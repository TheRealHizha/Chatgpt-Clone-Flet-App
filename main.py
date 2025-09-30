import json
import os
import uuid
from dataclasses import dataclass, asdict
from typing import List
import flet as ft
import g4f

# --- Config ---
DATA_FILE = "conversations.json"
DEFAULT_MODEL = "gpt-4o-mini"

# --- Data models ---
@dataclass
class Message:
    role: str  # 'user' or 'assistant' or 'system'
    content: str

@dataclass
class Conversation:
    id: str
    title: str
    messages: List[Message]


# --- Storage helpers ---
def load_conversations() -> List[Conversation]:
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            convos = []
            for c in data:
                convos.append(
                    Conversation(
                        id=c.get("id"),
                        title=c.get("title"),
                        messages=[Message(**m) for m in c.get("messages", [])]
                    )
                )
            return convos
    except Exception as e:
        print(f"Error loading conversations: {e}")
        return []

def save_conversations(convos: List[Conversation]):
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump([
                {"id": c.id, "title": c.title, "messages": [asdict(m) for m in c.messages]}
                for c in convos
            ], f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving conversations: {e}")


# --- UI helpers ---
def bubble(text: str, is_user: bool = False) -> ft.Container:
    return ft.Container(
        content=ft.Column([ft.Text(text, selectable=True)], tight=True),
        padding=ft.padding.symmetric(12, 10),
        alignment=ft.alignment.top_right if is_user else ft.alignment.top_left,
        bgcolor=ft.Colors.BLUE_700 if is_user else ft.Colors.GREY_900,
        border_radius=12,
        margin=ft.margin.only(bottom=10),
        width=520 if is_user else None,
    )


# --- Main App ---
def main(page: ft.Page):
    page.title = "Flet — Chat (g4f)"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 16
    page.window_width = 1100
    page.window_height = 720

    conversations: List[Conversation] = load_conversations()
    if not conversations:
        conversations.append(Conversation(id=str(uuid.uuid4()), title="New Chat", messages=[]))
    selected_index = 0

    convo_list = ft.ListView(expand=True, spacing=6, padding=6)
    messages_view = ft.ListView(expand=True, spacing=12, padding=12, auto_scroll=True)
    input_field = ft.TextField(expand=True, multiline=True, min_lines=1, max_lines=6, hint_text="Type a message...")
    send_button = ft.IconButton(icon=ft.Icons.SEND_ROUNDED)
    title_text = ft.Text("Chat — Flet + g4f", style=ft.TextStyle(size=20, weight="bold"))

    # Dynamically get available models from g4f to avoid invalid names
    try:
        available_models = sorted(list(g4f.models._all_models))
    except Exception as e:
        print(f"Failed to load g4f models: {e}")
        available_models = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gemini-pro", "claude-3-haiku"]

    model_dropdown = ft.Dropdown(
        width=220,
        value=DEFAULT_MODEL,
        options=[ft.dropdown.Option(model) for model in available_models],
        on_change=None  # will be set later
    )

    dark_toggle = ft.Switch(label="Dark mode", value=True)
    streaming_toggle = ft.Switch(label="Stream responses", value=True)

    def rebuild_convo_list():
        convo_list.controls.clear()
        for i, c in enumerate(conversations):
            btn = ft.ElevatedButton(
                c.title,
                width=260,
                on_click=lambda e, idx=i: select_convo(idx),
                style=ft.ButtonStyle(
                    color=ft.Colors.WHITE if i == selected_index else None,
                    bgcolor=ft.Colors.BLUE if i == selected_index else None
                )
            )
            convo_list.controls.append(btn)
        page.update()

    def rebuild_messages():
        messages_view.controls.clear()
        convo = conversations[selected_index]
        for m in convo.messages:
            is_user = m.role == "user"
            messages_view.controls.append(bubble(m.content, is_user))
        page.update()

    def select_convo(idx: int):
        nonlocal selected_index
        selected_index = idx
        rebuild_convo_list()
        rebuild_messages()

    def extract_response_content(response):
        """Extract text content from g4f response which can be various types"""
        try:
            if isinstance(response, str):
                return response
            elif hasattr(response, "content"):
                return str(response.content)
            elif hasattr(response, "__iter__"):
                # Handle non-streaming iterable response
                chunks = []
                for chunk in response:
                    if isinstance(chunk, str):
                        chunks.append(chunk)
                    elif hasattr(chunk, "content"):
                        chunks.append(chunk.content)
                    else:
                        chunks.append(str(chunk))
                return "".join(chunks)
            else:
                return str(response)
        except Exception:
            return "[Error: Could not parse response]"

    def stream_response(prompt: str):
        conv = conversations[selected_index]
        conv.messages.append(Message(role="user", content=prompt))
        save_conversations(conversations)
        rebuild_messages()

        assistant_msg_content = ""
        assistant_bubble = bubble(assistant_msg_content, False)
        messages_view.controls.append(assistant_bubble)
        page.update()

        try:
            model = model_dropdown.value or DEFAULT_MODEL

            # Try streaming if enabled
            if streaming_toggle.value:
                try:
                    response = g4f.ChatCompletion.create(
                        model=model,
                        messages=[{"role": m.role, "content": m.content} for m in conv.messages],
                        stream=True,
                    )
                    for chunk in response:
                        if isinstance(chunk, str):
                            assistant_msg_content += chunk
                        elif hasattr(chunk, "content"):
                            assistant_msg_content += chunk.content
                        else:
                            assistant_msg_content += str(chunk)
                        assistant_bubble.content.controls[0].value = assistant_msg_content
                        page.update()
                except Exception as e:
                    print(f"Streaming failed for {model}: {e}")
                    # Fallback to non-streaming
                    response = g4f.ChatCompletion.create(
                        model=model,
                        messages=[{"role": m.role, "content": m.content} for m in conv.messages],
                        stream=False,
                    )
                    assistant_msg_content = extract_response_content(response)
                    assistant_bubble.content.controls[0].value = assistant_msg_content
                    page.update()
            else:
                # Non-streaming mode
                response = g4f.ChatCompletion.create(
                    model=model,
                    messages=[{"role": m.role, "content": m.content} for m in conv.messages],
                    stream=False,
                )
                assistant_msg_content = extract_response_content(response)
                assistant_bubble.content.controls[0].value = assistant_msg_content
                page.update()

            # Save final assistant message
            conv.messages.append(Message(role="assistant", content=assistant_msg_content))
            save_conversations(conversations)

        except Exception as e:
            error_text = f"[Error: Cannot get response from '{model}'. {str(e)}]"
            assistant_msg_content += error_text
            assistant_bubble.content.controls[0].value = assistant_msg_content
            conv.messages.append(Message(role="assistant", content=assistant_msg_content))
            save_conversations(conversations)
            page.update()

    def on_send_click(e=None):
        text = input_field.value.strip()
        if not text:
            return

        # Update title if first message
        if not conversations[selected_index].messages:
            conversations[selected_index].title = text[:30] + "..." if len(text) > 30 else text
            rebuild_convo_list()

        input_field.value = ""
        page.update()
        stream_response(text)

    send_button.on_click = on_send_click
    input_field.on_submit = on_send_click

    left_col = ft.Column([
        ft.Row([title_text], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        ft.Divider(height=1),
        ft.Container(
            ft.Column([ft.Text("Conversations", weight="bold"), convo_list]),
            expand=True
        ),
        ft.Row([ft.ElevatedButton("New chat", on_click=lambda e: new_chat())]),
    ], width=300, spacing=12)

    right_col = ft.Column([
        ft.Row([model_dropdown, streaming_toggle, dark_toggle], alignment=ft.MainAxisAlignment.START),
        ft.Divider(height=1),
        messages_view,
        ft.Row([input_field, send_button], alignment=ft.MainAxisAlignment.CENTER),
    ], expand=True)

    def new_chat():
        new_id = str(uuid.uuid4())
        conversations.append(Conversation(id=new_id, title=f"Chat {len(conversations)+1}", messages=[]))
        save_conversations(conversations)
        rebuild_convo_list()
        select_convo(len(conversations) - 1)

    def on_dark(e):
        page.theme_mode = ft.ThemeMode.DARK if dark_toggle.value else ft.ThemeMode.LIGHT
        page.update()

    dark_toggle.on_change = on_dark

    # Keyboard shortcut: Enter to send, Shift+Enter for new line
    def on_key(e: ft.KeyboardEvent):
        if e.key == "Enter" and not e.shift and input_field.focused:
            on_send_click()
    page.on_keyboard_event = on_key

    # Rebuild UI
    rebuild_convo_list()
    rebuild_messages()

    # Add main layout
    page.add(ft.Row([left_col, ft.VerticalDivider(width=1), right_col], expand=True))


if __name__ == "__main__":
    ft.app(target=main, view=ft.AppView.FLET_APP)
