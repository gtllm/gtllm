use dioxus::prelude::*;
use dioxus::core::spawn_forever;
#[cfg(feature = "desktop")]
use dioxus::desktop::{use_window, use_wry_event_handler, WindowCloseBehaviour, WindowEvent};

mod components;
mod utils;

use components::{
    Choice, Collaborative, Competitive, Header, NewChat, PvP, Settings as SettingsView, Sidebar, Standard,
};
use utils::{ActiveRunRecord, AppView, ArenaMessage, ChatHistory, ChatMode, ChatSession, InputSettings, Message, OpenRouterClient, RunStatus, Settings, Theme};
use std::collections::HashMap;
use std::sync::Arc;

const TAILWIND_CSS: Asset = asset!("/assets/tailwind.css");

fn main() {
    #[cfg(feature = "desktop")]
    {
        let config = if cfg!(debug_assertions) {
            dioxus::desktop::Config::new().with_close_behaviour(WindowCloseBehaviour::WindowHides)
        } else {
            dioxus::desktop::Config::new()
                .with_menu(None)
                .with_close_behaviour(WindowCloseBehaviour::WindowHides)
        };

        dioxus::LaunchBuilder::desktop()
            .with_cfg(config)
            .launch(App);
    }

    #[cfg(not(feature = "desktop"))]
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    let active_runs = use_signal(HashMap::<String, ActiveRunRecord>::new);
    use_context_provider(|| active_runs);

    #[cfg(feature = "desktop")]
    {
        let window = use_window();
        let mut shutdown_requested = use_signal(|| false);
        let mut active_runs_for_close = active_runs.clone();
        let window_for_close = window.clone();

        use_wry_event_handler(move |event, _| {
            if let dioxus::desktop::tao::event::Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } = event
            {
                if shutdown_requested() {
                    return;
                }

                let active_ids: Vec<String> = active_runs_for_close
                    .read()
                    .iter()
                    .filter(|(_, run)| matches!(run.status, RunStatus::Running | RunStatus::Cancelling))
                    .map(|(id, _)| id.clone())
                    .collect();

                if active_ids.is_empty() {
                    window_for_close.set_close_behavior(WindowCloseBehaviour::WindowCloses);
                    window_for_close.close();
                    return;
                }

                shutdown_requested.set(true);

                {
                    let mut runs = active_runs_for_close.write();
                    for run_id in &active_ids {
                        if let Some(run) = runs.get_mut(run_id) {
                            run.request_cancel();
                            run.status = RunStatus::Cancelling;
                        }
                    }
                }

                let mut active_runs_for_shutdown = active_runs_for_close.clone();
                let window_for_shutdown = window_for_close.clone();
                spawn_forever(async move {
                    let deadline = tokio::time::Instant::now()
                        + std::time::Duration::from_secs(3);

                    loop {
                        let active_left = active_runs_for_shutdown
                            .read()
                            .values()
                            .any(|run| matches!(run.status, RunStatus::Running | RunStatus::Cancelling));
                        if !active_left || tokio::time::Instant::now() >= deadline {
                            break;
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    }

                    {
                        let runs = active_runs_for_shutdown.read();
                        for run in runs.values() {
                            if matches!(run.status, RunStatus::Running | RunStatus::Cancelling) {
                                run.task.cancel();
                            }
                        }
                    }

                    window_for_shutdown.set_close_behavior(WindowCloseBehaviour::WindowCloses);
                    window_for_shutdown.close();
                });
            }
        });
    }

    // Load settings from disk on startup
    let mut app_settings = use_signal(|| {
        Settings::load().unwrap_or_else(|e| {
            eprintln!("Failed to load settings: {}", e);
            Settings::default()
        })
    });

    // Create OpenRouter client if API key exists
    let mut openrouter_client = use_signal(|| {
        app_settings
            .read()
            .get_api_key()
            .and_then(|key| match OpenRouterClient::new(key.to_string()) {
                Ok(client) => Some(Arc::new(client)),
                Err(e) => {
                    eprintln!("Failed to initialize OpenRouter client: {}", e);
                    None
                }
            })
    });

    // Theme state - load from settings
    let mut theme = use_signal(|| {
        let settings = app_settings.read();
        // Parse theme from settings, default to Dracula if parsing fails
        Theme::from_str(&settings.theme).unwrap_or(Theme::Dracula)
    });

    // Sidebar state (default to open)
    let sidebar_collapsed = use_signal(|| false);

    // Current view state
    let mut current_view = use_signal(|| AppView::NewChat);

    // Messages for standard chat
    let mut messages = use_signal(|| Vec::<Message>::new());

    // Messages for arena modes
    let mut arena_messages = use_signal(|| Vec::<ArenaMessage>::new());

    // Chat sessions - load from disk on startup (fast, no file I/O)
    let mut sessions = use_signal(|| {
        ChatHistory::list_sessions()
            .unwrap_or_else(|e| {
                eprintln!("Failed to load sessions: {}", e);
                Vec::new()
            })
    });
    let mut current_session = use_signal(|| None::<String>);

    // Input settings
    let mut input_settings = use_signal(|| InputSettings {
        ctrl_enter_submit: true,
    });

    // Global loading state
    let mut is_loading = use_signal(|| false);

    // Message counter for IDs
    let mut message_counter = use_signal(|| 0);

    // Handler for toggling dark/light mode
    let toggle_mode = move |_| {
        let current_theme = *theme.read();
        let new_theme = if current_theme.is_dark() {
            // Switch to first light theme
            Theme::Winter
        } else {
            // Switch to first dark theme
            Theme::Dracula
        };
        theme.set(new_theme);

        // Save theme to settings
        let mut settings = app_settings.write();
        settings.theme = new_theme.to_string_id().to_string();
        settings.theme_mode = if new_theme.is_dark() {
            utils::ThemeMode::Dark
        } else {
            utils::ThemeMode::Light
        };
        if let Err(e) = settings.save() {
            eprintln!("Failed to save theme settings: {}", e);
        }
    };

    // Handler for changing theme within mode
    let change_theme = move |new_theme: Theme| {
        theme.set(new_theme);

        // Save theme to settings
        let mut settings = app_settings.write();
        settings.theme = new_theme.to_string_id().to_string();
        settings.theme_mode = if new_theme.is_dark() {
            utils::ThemeMode::Dark
        } else {
            utils::ThemeMode::Light
        };
        if let Err(e) = settings.save() {
            eprintln!("Failed to save theme settings: {}", e);
        }
    };

    // Handler for saving API key
    let save_api_key = move |api_key: String| {
        let mut settings = app_settings.write();
        settings.set_api_key(api_key.clone());
        if let Err(e) = settings.save() {
            eprintln!("Failed to save API key: {}", e);
            return;
        }

        // Recreate OpenRouter client with new API key
        match OpenRouterClient::new(api_key) {
            Ok(client) => openrouter_client.set(Some(Arc::new(client))),
            Err(e) => {
                eprintln!("Failed to initialize OpenRouter client: {}", e);
                openrouter_client.set(None);
            }
        }
    };

    // Handler for selecting a session (no full list refresh — use in-memory list and verify file exists)
    let select_session = {
        let mut sessions_clone = sessions.clone();
        let mut current_session = current_session.clone();
        let mut current_view = current_view.clone();
        let mut messages = messages.clone();
        let mut arena_messages = arena_messages.clone();
        let mut is_loading = is_loading.clone();

        move |session_id: String| {
            is_loading.set(true);
            let session_id = session_id.clone();
            let mut sessions_clone = sessions_clone.clone();
            let mut current_session = current_session.clone();
            let mut current_view = current_view.clone();
            let mut messages = messages.clone();
            let mut arena_messages = arena_messages.clone();
            let mut is_loading = is_loading.clone();

            spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;

                let session_opt = sessions_clone.read().iter().find(|s| s.id == session_id).cloned();

                if let Some(session) = session_opt {
                    let session_path_exists = tokio::task::spawn_blocking(move || {
                        ChatHistory::session_path(&session_id).and_then(|p| {
                            if p.exists() { Ok(()) } else { Err("not found".into()) }
                        }).is_ok()
                    }).await.unwrap_or(false);

                    if session_path_exists {
                        current_session.set(Some(session.id.clone()));
                        current_view.set(AppView::ChatMode(session.mode));
                        messages.write().clear();
                        arena_messages.write().clear();
                    } else {
                        eprintln!("Session file not found: {}", session.id);
                        sessions_clone.write().retain(|s| s.id != session.id);
                    }
                } else {
                    eprintln!("Session not found in list");
                }

                is_loading.set(false);
            });
        }
    };

    // Handler for creating new chat
    let new_chat = {
        let mut sessions_clone = sessions.clone();
        let mut current_view = current_view.clone();
        let mut messages = messages.clone();
        let mut arena_messages = arena_messages.clone();
        let mut current_session = current_session.clone();
        let mut is_loading = is_loading.clone();

        move |_| {
            is_loading.set(true);
            let mut sessions_clone = sessions_clone.clone();
            let mut current_view = current_view.clone();
            let mut messages = messages.clone();
            let mut arena_messages = arena_messages.clone();
            let mut current_session = current_session.clone();
            let mut is_loading = is_loading.clone();

            spawn(async move {
                // Small delay to ensure UI updates
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;

                current_view.set(AppView::NewChat);
                messages.write().clear();
                arena_messages.write().clear();
                current_session.set(None);
                
                // Refresh sessions list to pick up any new or renamed sessions
                let sessions_result = tokio::task::spawn_blocking(|| ChatHistory::list_sessions()).await;
                
                match sessions_result {
                    Ok(Ok(new_sessions)) => {
                        sessions_clone.set(new_sessions);
                    }
                    Ok(Err(e)) => {
                        eprintln!("Failed to refresh sessions: {}", e);
                    }
                    Err(e) => {
                        eprintln!("Task join error: {}", e);
                    }
                }
                
                is_loading.set(false);
            });
        }
    };

    // Handler for mode selection from NewChat view.
    // Do not add to sessions here — session is added only on first save (when there is content).
    let select_mode = move |mode: ChatMode| {
        let timestamp = ChatHistory::format_timestamp();
        let title = format!("{} Chat", mode.name());
        let session_id = ChatHistory::generate_session_id(mode, &timestamp, &title);
        current_session.set(Some(session_id));
        current_view.set(AppView::ChatMode(mode));
        messages.write().clear();
        arena_messages.write().clear();
    };

    // Called by mode components when they save a session for the first time (with content).
    // Ensures the session appears in the sidebar only after real activity.
    let on_session_saved = move |session: ChatSession| {
        if !sessions.read().iter().any(|s| s.id == session.id) {
            sessions.write().push(session);
        }
    };

    // Handler for opening settings
    let open_settings = move |_| {
        current_view.set(AppView::Settings);
    };

    // Handler for closing settings
    let close_settings = move |_| {
        if let Some(session_id) = current_session.read().clone() {
            if let Some(session) = sessions.read().iter().find(|s| s.id == session_id) {
                current_view.set(AppView::ChatMode(session.mode));
            } else {
                current_view.set(AppView::NewChat);
            }
        } else {
            current_view.set(AppView::NewChat);
        }
    };

    // Handler for input settings change
    let change_input_settings = move |new_settings: InputSettings| {
        input_settings.set(new_settings);
    };

    // Handler for sending messages in standard mode
    let mut send_message = move |text: String| {
        let id = *message_counter.read();
        message_counter.set(id + 1);

        // Add user message
        messages.write().push(Message {
            id,
            content: text.clone(),
            sender: "You".to_string(),
            is_user: true,
            timestamp: "Just now".to_string(),
        });

        // Simulate bot response
        let bot_id = id + 1;
        message_counter.set(bot_id + 1);
        messages.write().push(Message {
            id: bot_id,
            content: format!("This is a simulated response to: '{}'", text),
            sender: "Assistant".to_string(),
            is_user: false,
            timestamp: "Just now".to_string(),
        });
    };

    // Handler for sending messages in arena mode
    let mut send_arena_message = move |text: String, mode: ChatMode| {
        let base_id = *message_counter.read();

        match mode {
            ChatMode::PvP => {
                // Add user message first
                messages.write().push(Message {
                    id: base_id,
                    content: text.clone(),
                    sender: "You".to_string(),
                    is_user: true,
                    timestamp: "Just now".to_string(),
                });

                // Add responses from Bot 1, Bot 2, and Moderator
                messages.write().push(Message {
                    id: base_id + 1,
                    content: format!("Bot 1's response to: '{}'", text),
                    sender: "Bot 1".to_string(),
                    is_user: false,
                    timestamp: "Just now".to_string(),
                });

                messages.write().push(Message {
                    id: base_id + 2,
                    content: format!("Bot 2's response to: '{}'", text),
                    sender: "Bot 2".to_string(),
                    is_user: false,
                    timestamp: "Just now".to_string(),
                });

                messages.write().push(Message {
                    id: base_id + 3,
                    content: "As moderator, I find Bot 1's response more comprehensive."
                        .to_string(),
                    sender: "Moderator".to_string(),
                    is_user: false,
                    timestamp: "Just now".to_string(),
                });

                message_counter.set(base_id + 4);
            }
            ChatMode::Collaborative => {
                // Add user message
                messages.write().push(Message {
                    id: base_id,
                    content: text.clone(),
                    sender: "You".to_string(),
                    is_user: true,
                    timestamp: "Just now".to_string(),
                });

                // Add collaborative responses
                for i in 0..3 {
                    messages.write().push(Message {
                        id: base_id + i + 1,
                        content: format!("Bot {} contributes to the solution...", i + 1),
                        sender: format!("Bot {}", i + 1),
                        is_user: false,
                        timestamp: "Just now".to_string(),
                    });
                }
                message_counter.set(base_id + 4);
            }
            ChatMode::Competitive => {
                // Add user message
                messages.write().push(Message {
                    id: base_id,
                    content: text.clone(),
                    sender: "You".to_string(),
                    is_user: true,
                    timestamp: "Just now".to_string(),
                });

                // Add competitive responses
                for i in 0..3 {
                    messages.write().push(Message {
                        id: base_id + i + 1,
                        content: format!(
                            "Bot {}'s competitive solution to: '{}' (votes: {})",
                            i + 1,
                            text,
                            i % 3 + 1
                        ),
                        sender: format!("Bot {}", i + 1),
                        is_user: false,
                        timestamp: "Just now".to_string(),
                    });
                }
                message_counter.set(base_id + 4);
            }
            ChatMode::LLMChoice => {
                // Add user message
                messages.write().push(Message {
                    id: base_id,
                    content: text,
                    sender: "You".to_string(),
                    is_user: true,
                    timestamp: "Just now".to_string(),
                });

                messages.write().push(Message {
                    id: base_id + 1,
                    content: "LLMs have chosen to COLLABORATE on this task.".to_string(),
                    sender: "System".to_string(),
                    is_user: false,
                    timestamp: "Just now".to_string(),
                });
                message_counter.set(base_id + 2);
            }
            _ => {}
        }
    };

    let theme_val = *theme.read();
    let data_theme = theme_val.data_theme();

    rsx! {
        document::Link { rel: "stylesheet", href: TAILWIND_CSS }

        div {
            "data-theme": "{data_theme}",
            class: "font-inter antialiased bg-[var(--color-base-100)] text-[var(--color-base-content)]",

            div {
                class: "flex h-screen overflow-hidden relative",

                // Sidebar
                Sidebar {
                    theme,
                    sessions,
                    current_session,
                    collapsed: sidebar_collapsed,
                    on_new_chat: new_chat,
                    on_select_session: select_session,
                }

                // Main content area
                div {
                    class: "relative flex flex-col flex-1 overflow-hidden min-h-0",

                    // Header
                    Header {
                        theme,
                        on_toggle_mode: toggle_mode,
                        on_theme_change: change_theme,
                        on_settings: open_settings,
                        client: openrouter_client,
                    }

                    // Main content based on current view
                    main {
                        class: "grow min-h-0",

                        match *current_view.read() {
                            AppView::NewChat => rsx! {
                                NewChat {
                                    theme,
                                    app_settings,
                                    on_mode_select: select_mode,
                                    on_open_settings: open_settings,
                                }
                            },
                            AppView::ChatMode(mode) => {
                                let session_id = current_session.read().clone();
                                match mode {
                                    ChatMode::Standard => rsx! {
                                        Standard {
                                            theme,
                                            client: openrouter_client.read().clone(),
                                            input_settings,
                                            session_id,
                                            on_session_saved,
                                        }
                                    },
                                    ChatMode::PvP => rsx! {
                                        PvP {
                                            theme,
                                            client: openrouter_client.read().clone(),
                                            input_settings,
                                            session_id,
                                            on_session_saved,
                                        }
                                    },
                                    ChatMode::Collaborative => rsx! {
                                        Collaborative {
                                            theme,
                                            client: openrouter_client.read().clone(),
                                            input_settings,
                                            session_id,
                                            on_session_saved,
                                        }
                                    },
                                    ChatMode::Competitive => rsx! {
                                        Competitive {
                                            theme,
                                            client: openrouter_client.read().clone(),
                                            input_settings,
                                            session_id,
                                            on_session_saved,
                                        }
                                    },
                                    ChatMode::LLMChoice => rsx! {
                                        Choice {
                                            theme,
                                            client: openrouter_client.read().clone(),
                                            input_settings,
                                            session_id,
                                            on_session_saved,
                                        }
                                    },
                                }
                            },
                            AppView::Settings => rsx! {
                                SettingsView {
                                    theme,
                                    app_settings,
                                    input_settings,
                                    on_settings_change: change_input_settings,
                                    on_api_key_save: save_api_key,
                                    on_close: close_settings,
                                }
                            },
                        }
                    }
                }
                
                // Loading Overlay
                if *is_loading.read() {
                    div {
                        class: "absolute inset-0 bg-black/50 z-50 flex items-center justify-center backdrop-blur-sm",
                        div {
                            class: "bg-[var(--color-base-100)] p-6 rounded-xl shadow-2xl flex flex-col items-center gap-4 border border-[var(--color-base-300)]",
                            div {
                                class: "w-12 h-12 border-4 border-[var(--color-primary)] border-t-transparent rounded-full animate-spin"
                            }
                            div {
                                class: "text-lg font-medium text-[var(--color-base-content)]",
                                "Loading..."
                            }
                        }
                    }
                }
            }
        }
    }
}
