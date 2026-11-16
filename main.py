"""Telegram-бот для игры «Шпион».

Админ создаёт комнату (кнопка «Новая игра»), получает ссылку-приглашение, выбирает тему,
друзья заходят по ссылке. Слово по теме подбирает Gemini, роли раздаёт код.
Игрок, который начинает обсуждение, выбирается случайно среди всех (шпион тоже может быть первым).

Для соло-теста есть «фейковые» игроки: их роли бот присылает админу.
"""

import asyncio
import html
import json
import logging
import os
import random
import secrets
from dataclasses import dataclass, field

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

load_dotenv()

# ---------------------------------------------------------------- настройки

BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Модели пробуются по порядку: если первая перегружена или недоступна, берётся следующая.
# Если везде 404 — названия снова поменялись, обнови список.
GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-2.5-flash"]
GEMINI_TIMEOUT = 30  # секунд на один запрос; если Gemini молчит, мы увидим ошибку
RETRY_CODES = {429, 500, 502, 503, 504}  # временные сбои: повторяем запрос
RETRIES_PER_MODEL = 3
WORDS_PER_REQUEST = 40  # сколько слов просим за один запрос и кладём в кэш

SPY_KNOWS_THEME = True  # шпион знает тему (но не слово)
MIN_PLAYERS = 3
MAX_PLAYERS = 20
MAX_SPIES = 5

FAKE_NAMES = ["Вася", "Петя", "Оля", "Маша", "Коля", "Даша", "Игорь", "Лена",
              "Саша", "Женя", "Антон", "Катя"]
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # без похожих символов

# ------------------------------------------------------------------- данные


@dataclass
class Room:
    code: str
    admin_id: int
    players: dict[int, str] = field(default_factory=dict)  # id -> имя
    theme: str = ""
    spies_count: int = 1
    playing: bool = False
    busy: bool = False  # пока идёт запрос к нейронке
    word: str = ""
    spies: set[int] = field(default_factory=set)
    starter: int | None = None  # кто начинает обсуждение (может быть и шпионом)
    panel_message_id: int | None = None
    next_fake_id: int = -1
    banned: set[int] = field(default_factory=set)  # кикнутые не могут вернуться


ROOMS: dict[str, Room] = {}
USER_ROOM: dict[int, str] = {}  # id игрока -> код комнаты
WORD_CACHE: dict[str, list[str]] = {}  # тема -> список слов
USED_WORDS: dict[str, set[str]] = {}  # тема -> уже выпадавшие слова

BOT_USERNAME = ""
gemini: genai.Client | None = None

router = Router()
router.message.filter(F.chat.type == "private")


class Setup(StatesGroup):
    theme = State()


# ------------------------------------------------------------------ helpers


NEW_GAME_BTN = "🎮 Новая игра"


def main_menu() -> ReplyKeyboardMarkup:
    """Постоянная кнопка внизу чата."""
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=NEW_GAME_BTN)]],
        resize_keyboard=True,
        is_persistent=True,
    )


def esc(text: str) -> str:
    return html.escape(text)


def is_fake(pid: int) -> bool:
    return pid < 0


def new_code() -> str:
    while True:
        code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(6))
        if code not in ROOMS:
            return code


def get_user_room(user_id: int) -> Room | None:
    return ROOMS.get(USER_ROOM.get(user_id, ""))


def get_admin_room(user_id: int) -> Room | None:
    room = get_user_room(user_id)
    return room if room and room.admin_id == user_id else None


# ---------------------------------------------------------------- нейронка


async def generate_with_fallback(prompt: str):
    """Запрос к Gemini с повторами при перегрузке и переходом на запасные модели."""
    last_error: Exception | None = None
    for model in GEMINI_MODELS:
        for attempt in range(1, RETRIES_PER_MODEL + 1):
            try:
                return await asyncio.wait_for(
                    gemini.aio.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=1.0,
                        ),
                    ),
                    timeout=GEMINI_TIMEOUT,
                )
            except genai_errors.APIError as e:
                last_error = e
                if e.code in RETRY_CODES and attempt < RETRIES_PER_MODEL:
                    logging.warning(
                        "%s: ошибка %s, попытка %d/%d, жду…",
                        model, e.code, attempt, RETRIES_PER_MODEL,
                    )
                    await asyncio.sleep(2 * attempt)
                    continue
                logging.warning("Модель %s не сработала (%s), пробую следующую", model, e.code)
                break
            except asyncio.TimeoutError as e:
                last_error = e
                logging.warning("Модель %s не ответила за %d с, пробую следующую", model, GEMINI_TIMEOUT)
                break
    raise last_error or RuntimeError("Не удалось получить ответ ни от одной модели")


async def fetch_words(theme: str) -> list[str]:
    prompt = (
        f"Тема игры в шпиона: «{theme}».\n"
        f"Составь список из {WORDS_PER_REQUEST} разных слов или коротких названий "
        "по этой теме. Слова должны быть конкретными (не общие категории), "
        "знакомыми большинству людей, которые разбираются в теме, и разной "
        "степени очевидности. Отвечай на языке темы. "
        "Ответ — строго JSON-массив строк, без пояснений."
    )
    response = await generate_with_fallback(prompt)
    data = json.loads(response.text)
    if not isinstance(data, list):
        raise ValueError(f"Нейронка вернула не список: {response.text[:200]}")
    words: list[str] = []
    for item in data:
        word = str(item).strip()
        if word and word not in words:
            words.append(word)
    if len(words) < 5:
        raise ValueError(f"Слишком мало слов в ответе: {words}")
    return words


async def pick_word(theme: str) -> str:
    """Случайное слово по теме. Случайность — на стороне кода, не нейронки."""
    key = theme.casefold()
    used = USED_WORDS.setdefault(key, set())
    words = WORD_CACHE.get(key)
    candidates = [w for w in (words or []) if w not in used]
    if not candidates:
        words = await fetch_words(theme)
        WORD_CACHE[key] = words
        used.clear()
        candidates = words
    word = random.choice(candidates)
    used.add(word)
    return word


# ------------------------------------------------------------ панель админа


def render_panel(room: Room) -> tuple[str, InlineKeyboardMarkup]:
    link = f"https://t.me/{BOT_USERNAME}?start=room_{room.code}"
    lines = []
    for i, (pid, name) in enumerate(room.players.items(), 1):
        mark = " 👑" if pid == room.admin_id else ""
        mark += " 🧪" if is_fake(pid) else ""
        lines.append(f"{i}. {esc(name)}{mark}")
    players_text = "\n".join(lines)

    kb = InlineKeyboardBuilder()
    if room.playing:
        starter_name = esc(room.players.get(room.starter, "?"))
        text = (
            f"🕵️ <b>Комната {room.code}</b> — идёт раунд\n\n"
            f"Тема: <b>{esc(room.theme)}</b>\n"
            f"Шпионов: {room.spies_count}\n"
            f"Начинает: <b>{starter_name}</b>\n\n"
            f"Игроки ({len(room.players)}):\n{players_text}\n\n"
            "Когда закончите обсуждение, нажми «Завершить»."
        )
        kb.button(text="🏁 Завершить раунд", callback_data="adm:finish")
        kb.adjust(1)
    else:
        theme = f"<b>{esc(room.theme)}</b>" if room.theme else "<i>не выбрана</i>"
        text = (
            f"🕵️ <b>Комната {room.code}</b>\n\n"
            f"Ссылка для друзей:\n<code>{link}</code>\n\n"
            f"Тема: {theme}\n"
            f"Шпионов: {room.spies_count}\n\n"
            f"Игроки ({len(room.players)}):\n{players_text}"
        )
        kb.button(text="🎯 Выбрать тему", callback_data="adm:theme")
        kb.button(text="➖ Шпион", callback_data="adm:spy_minus")
        kb.button(text="➕ Шпион", callback_data="adm:spy_plus")
        kb.button(text="▶️ Начать раунд", callback_data="adm:start")
        kb.button(text="🧪 + Фейк", callback_data="adm:fake_add")
        kb.button(text="🧪 − Фейк", callback_data="adm:fake_del")
        kb.button(text="👢 Кикнуть игрока", callback_data="adm:kick")
        kb.button(text="❌ Закрыть комнату", callback_data="adm:close")
        kb.adjust(1, 2, 1, 2, 1, 1)
    return text, kb.as_markup()


async def send_panel(bot: Bot, room: Room) -> None:
    text, kb = render_panel(room)
    msg = await bot.send_message(room.admin_id, text, reply_markup=kb)
    room.panel_message_id = msg.message_id


async def refresh_panel(bot: Bot, room: Room, status: str | None = None) -> None:
    if room.panel_message_id is None:
        return
    if status:
        text, kb = status, None
    else:
        text, kb = render_panel(room)
    try:
        await bot.edit_message_text(
            text=text,
            chat_id=room.admin_id,
            message_id=room.panel_message_id,
            reply_markup=kb,
        )
    except TelegramBadRequest as e:
        if "message is not modified" not in str(e):
            logging.warning("Не удалось обновить панель: %s", e)


# ------------------------------------------------------------ отправка ролей


async def deliver(bot: Bot, room: Room, pid: int, text: str) -> None:
    """Реальному игроку — в личку, фейковому — админу с пометкой."""
    try:
        if is_fake(pid):
            await bot.send_message(
                room.admin_id, f"🧪 <b>[{esc(room.players[pid])}]</b>\n{text}"
            )
        else:
            await bot.send_message(pid, text)
    except TelegramAPIError as e:
        logging.warning("Не доставлено игроку %s: %s", pid, e)
        if not is_fake(pid):
            await bot.send_message(
                room.admin_id,
                f"⚠️ Не смог написать игроку {esc(room.players.get(pid, str(pid)))}: "
                f"{esc(str(e))}",
            )


# ------------------------------------------------------- вход/выход из комнат


async def close_room(bot: Bot, room: Room, notify: bool = True) -> None:
    ROOMS.pop(room.code, None)
    for pid in list(room.players):
        USER_ROOM.pop(pid, None)
        if notify and not is_fake(pid) and pid != room.admin_id:
            try:
                await bot.send_message(pid, "Комната закрыта админом.")
            except TelegramAPIError:
                pass


async def detach_user(bot: Bot, user_id: int) -> Room | None:
    """Убирает игрока из текущей комнаты. Возвращает комнату, панель которой надо обновить."""
    room = get_user_room(user_id)
    USER_ROOM.pop(user_id, None)
    if room is None:
        return None
    if room.admin_id == user_id:
        await close_room(bot, room)
        return None
    room.players.pop(user_id, None)
    return room


# ---------------------------------------------------------------- команды


@router.message(CommandStart(deep_link=True))
async def cmd_start_deeplink(message: Message, command: CommandObject, bot: Bot) -> None:
    payload = command.args or ""
    if not payload.startswith("room_"):
        await cmd_start(message)
        return

    uid = message.from_user.id
    room = ROOMS.get(payload.removeprefix("room_").upper())
    if room is None:
        await message.answer(
            "Такой комнаты нет (возможно, бота перезапускали). "
            "Попроси админа создать новую."
        )
        return
    if uid in room.players:
        await message.answer("Ты уже в этой комнате ✅", reply_markup=main_menu())
        return
    if uid in room.banned:
        await message.answer("Админ убрал тебя из этой комнаты.")
        return
    if room.playing:
        await message.answer(
            "В комнате сейчас идёт раунд. Дождись конца и перейди по ссылке снова."
        )
        return
    if len(room.players) >= MAX_PLAYERS:
        await message.answer("В комнате уже максимум игроков.")
        return

    prev = await detach_user(bot, uid)
    room.players[uid] = message.from_user.full_name
    USER_ROOM[uid] = room.code
    await message.answer(
        f"Ты в комнате <b>{room.code}</b> ✅\n"
        "Жди, пока админ начнёт раунд. Роль придёт сюда, в этот чат.",
        reply_markup=main_menu(),
    )
    await refresh_panel(bot, room)
    if prev is not None:
        await refresh_panel(bot, prev)


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Привет! Это бот для игры в шпиона 🕵️\n\n"
        f"Нажми «{NEW_GAME_BTN}» внизу, чтобы создать комнату и получить "
        "ссылку для друзей.\n"
        "/leave — выйти из комнаты\n\n"
        "Чтобы зайти в чужую комнату, просто перейди по ссылке от админа.",
        reply_markup=main_menu(),
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await cmd_start(message)


@router.message(F.text == NEW_GAME_BTN)
@router.message(Command("newgame"))
async def cmd_newgame(message: Message, bot: Bot, state: FSMContext) -> None:
    await state.clear()
    uid = message.from_user.id
    prev = await detach_user(bot, uid)  # закроет старую комнату, если ты был её админом
    room = Room(code=new_code(), admin_id=uid)
    room.players[uid] = message.from_user.full_name
    ROOMS[room.code] = room
    USER_ROOM[uid] = room.code
    await send_panel(bot, room)
    if prev is not None:
        await refresh_panel(bot, prev)


@router.message(Command("leave"))
async def cmd_leave(message: Message, bot: Bot) -> None:
    uid = message.from_user.id
    room = get_user_room(uid)
    if room is None:
        await message.answer("Ты сейчас не в комнате.")
    elif room.admin_id == uid:
        await message.answer("Ты админ. Закрыть комнату можно кнопкой в панели.")
    elif room.playing:
        await message.answer("Сейчас идёт раунд — выйти можно после его завершения.")
    else:
        await detach_user(bot, uid)
        await message.answer("Ты вышел из комнаты.")
        await refresh_panel(bot, room)


@router.message(Setup.theme, F.text, ~F.text.startswith("/"))
async def got_theme(message: Message, bot: Bot, state: FSMContext) -> None:
    room = get_admin_room(message.from_user.id)
    await state.clear()
    if room is None:
        await message.answer(f"Комната не найдена. Нажми «{NEW_GAME_BTN}».")
        return
    room.theme = message.text.strip()[:80]
    await message.answer(f"Тема: <b>{esc(room.theme)}</b> ✅")
    await refresh_panel(bot, room)


# ----------------------------------------------------------- кнопки админа


@router.callback_query(F.data.startswith("adm:"))
async def on_admin_action(cb: CallbackQuery, bot: Bot, state: FSMContext) -> None:
    room = get_admin_room(cb.from_user.id)
    if room is None:
        await cb.answer(f"Комната не найдена. Нажми «{NEW_GAME_BTN}».", show_alert=True)
        return
    if room.busy:
        await cb.answer("Подожди, подбираю слово…")
        return

    action = cb.data.split(":", 1)[1]

    if action == "finish":
        if not room.playing:
            await cb.answer("Раунд уже завершён.")
            return
        await finish_round(cb, bot, room)
        return

    if room.playing:
        await cb.answer("Сначала заверши текущий раунд.", show_alert=True)
        return

    if action == "theme":
        await state.set_state(Setup.theme)
        await cb.message.answer(
            "Напиши тему одним сообщением, например: <i>герои Dota 2</i>, "
            "<i>еда в Италии</i>, <i>профессии</i>."
        )
        await cb.answer()

    elif action == "spy_plus":
        room.spies_count = min(MAX_SPIES, room.spies_count + 1)
        await cb.answer()
        await refresh_panel(bot, room)

    elif action == "spy_minus":
        room.spies_count = max(1, room.spies_count - 1)
        await cb.answer()
        await refresh_panel(bot, room)

    elif action == "fake_add":
        if len(room.players) >= MAX_PLAYERS:
            await cb.answer("Больше нельзя.", show_alert=True)
            return
        taken = set(room.players.values())
        free = [n for n in FAKE_NAMES if n not in taken]
        name = free[0] if free else f"Фейк {abs(room.next_fake_id)}"
        room.players[room.next_fake_id] = name
        room.next_fake_id -= 1
        await cb.answer(f"Добавлен {name}")
        await refresh_panel(bot, room)

    elif action == "fake_del":
        fakes = [pid for pid in room.players if is_fake(pid)]
        if not fakes:
            await cb.answer("Фейков нет.", show_alert=True)
            return
        room.players.pop(min(fakes))  # последний добавленный
        await cb.answer()
        await refresh_panel(bot, room)

    elif action == "kick":
        kickable = [(p, n) for p, n in room.players.items() if p != room.admin_id]
        if not kickable:
            await cb.answer("Некого кикать.", show_alert=True)
            return
        kb = InlineKeyboardBuilder()
        for pid, name in kickable:
            label = f"{name} 🧪" if is_fake(pid) else name
            kb.button(text=f"👢 {label}", callback_data=f"kick:{pid}")
        kb.button(text="↩️ Назад", callback_data="kick:back")
        kb.adjust(1)
        await cb.answer()
        try:
            await cb.message.edit_text("Кого убрать из комнаты?", reply_markup=kb.as_markup())
        except TelegramBadRequest as e:
            logging.warning("Не удалось показать список для кика: %s", e)

    elif action == "close":
        await close_room(bot, room)
        await cb.answer()
        try:
            await cb.message.edit_text(f"Комната закрыта. Новая: «{NEW_GAME_BTN}»")
        except TelegramAPIError:
            pass

    elif action == "start":
        await start_round(cb, bot, room)

    else:
        await cb.answer()


@router.callback_query(F.data.startswith("kick:"))
async def on_kick(cb: CallbackQuery, bot: Bot) -> None:
    room = get_admin_room(cb.from_user.id)
    if room is None:
        await cb.answer(f"Комната не найдена. Нажми «{NEW_GAME_BTN}».", show_alert=True)
        return
    arg = cb.data.split(":", 1)[1]
    if room.playing or room.busy or arg == "back":
        await cb.answer()
        await refresh_panel(bot, room)
        return
    try:
        pid = int(arg)
    except ValueError:
        await cb.answer()
        return
    if pid == room.admin_id or pid not in room.players:
        await cb.answer("Этого игрока уже нет в комнате.")
        await refresh_panel(bot, room)
        return

    name = room.players.pop(pid)
    if not is_fake(pid):
        room.banned.add(pid)  # по старой ссылке уже не вернётся
        USER_ROOM.pop(pid, None)
        try:
            await bot.send_message(pid, "Админ убрал тебя из комнаты.")
        except TelegramAPIError:
            pass
    await cb.answer(f"{name} удалён")
    await refresh_panel(bot, room)


async def start_round(cb: CallbackQuery, bot: Bot, room: Room) -> None:
    n = len(room.players)
    if not room.theme:
        await cb.answer("Сначала выбери тему.", show_alert=True)
        return
    if n < MIN_PLAYERS:
        await cb.answer(
            f"Нужно минимум {MIN_PLAYERS} игрока. Добавь фейков кнопкой 🧪.",
            show_alert=True,
        )
        return
    if room.spies_count * 2 >= n:
        await cb.answer("Слишком много шпионов для такого числа игроков.", show_alert=True)
        return

    room.busy = True
    await cb.answer()
    await refresh_panel(bot, room, status="⏳ Подбираю слово у нейронки…")
    try:
        word = await pick_word(room.theme)
    except Exception as e:  # показываем ошибку явно, а не молчим
        logging.exception("Ошибка при получении слова")
        room.busy = False
        await refresh_panel(bot, room)
        await cb.message.answer(
            "⚠️ Не получилось получить слово от нейронки:\n"
            f"<code>{esc(type(e).__name__)}: {esc(str(e))[:500]}</code>"
        )
        return
    room.busy = False

    room.word = word
    room.spies = set(random.sample(list(room.players), room.spies_count))
    # Начинающий выбирается среди всех игроков, включая шпионов.
    room.starter = random.choice(list(room.players))
    room.playing = True

    starter_line = f"\n\n🎤 Обсуждение начинает: <b>{esc(room.players[room.starter])}</b>"

    for pid in room.players:
        if pid in room.spies:
            text = "🕵️ <b>Ты шпион!</b>"
            if SPY_KNOWS_THEME:
                text += f"\nТема: <b>{esc(room.theme)}</b>"
            text += "\nСлово ты не знаешь — слушай остальных и не выдай себя."
        else:
            text = (
                f"🔎 Тема: <b>{esc(room.theme)}</b>\n"
                f"Слово: <b>{esc(room.word)}</b>\n\n"
                "Обсуждайте и вычисляйте шпиона."
            )
        text += starter_line
        if pid == room.starter:
            text += " (это ты!)"
        await deliver(bot, room, pid, text)

    await refresh_panel(bot, room)


async def finish_round(cb: CallbackQuery, bot: Bot, room: Room) -> None:
    spies = ", ".join(esc(room.players[p]) for p in room.spies if p in room.players)
    label = "Шпион" if len(room.spies) == 1 else "Шпионы"
    text = (
        "🏁 <b>Раунд окончен</b>\n\n"
        f"Слово: <b>{esc(room.word)}</b>\n"
        f"{label}: <b>{spies}</b>"
    )
    for pid in room.players:
        if not is_fake(pid):
            try:
                await bot.send_message(pid, text)
            except TelegramAPIError as e:
                logging.warning("Не доставлено игроку %s: %s", pid, e)

    room.playing = False
    room.word = ""
    room.spies = set()
    room.starter = None
    await cb.answer()
    try:  # у старой панели убираем кнопки
        await cb.message.edit_reply_markup(reply_markup=None)
    except TelegramAPIError:
        pass
    await send_panel(bot, room)  # новая панель — внизу чата, готова к новому раунду


# --------------------------------------------------------------------- main


async def main() -> None:
    global BOT_USERNAME, gemini
    logging.basicConfig(level=logging.INFO)

    if not BOT_TOKEN or not GEMINI_API_KEY:
        raise SystemExit(
            "Заполни файл .env: нужны BOT_TOKEN=... и GEMINI_API_KEY=... "
            "(без пробелов и кавычек)."
        )

    gemini = genai.Client(api_key=GEMINI_API_KEY)
    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(
            parse_mode=ParseMode.HTML, link_preview_is_disabled=True
        ),
    )
    BOT_USERNAME = (await bot.get_me()).username
    logging.info("Бот запущен: @%s", BOT_USERNAME)

    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())