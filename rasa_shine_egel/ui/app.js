const form = document.getElementById("composer");
const input = document.getElementById("composer-input");
const messages = document.getElementById("messages");
const statusPill = document.getElementById("status-pill");
const sendButton = form.querySelector(".send");
const navItems = document.querySelectorAll(".nav-item[data-view]");
const quickButtons = document.querySelectorAll(".pill");
const newChatButtons = document.querySelectorAll("[data-action=\"new-chat\"]");

const endpoint = form.dataset.endpoint;
const baseUrl = endpoint.replace(/\/webhooks\/rest\/webhook\/?$/, "");

const views = document.querySelectorAll(".view");
const historyList = document.getElementById("history-list");
const endpointLabel = document.getElementById("endpoint-label");
const clearHistoryButton = document.getElementById("clear-history");

const chatsKey = "rasa_chats";
const currentChatKey = "rasa_current_chat";
const legacyHistoryKey = "rasa_history";

let chats = [];
let currentChatId = null;
let senderId = null;

const loadChats = () => {
  try {
    chats = JSON.parse(localStorage.getItem(chatsKey) || "[]");
  } catch {
    chats = [];
  }

  if (!chats.length) {
    try {
      const legacy = JSON.parse(localStorage.getItem(legacyHistoryKey) || "[]");
      if (legacy.length) {
        const id = crypto.randomUUID();
        chats = [
          {
            id,
            title: "Imported chat",
            createdAt: Date.now(),
            messages: legacy.map((item) => ({
              id: item.id || crypto.randomUUID(),
              role: item.role || "user",
              text: item.text || "",
              ts: Date.now(),
            })),
          },
        ];
        localStorage.removeItem(legacyHistoryKey);
        localStorage.setItem(chatsKey, JSON.stringify(chats));
        localStorage.setItem(currentChatKey, id);
      }
    } catch {
      // ignore legacy parsing errors
    }
  }
};

const saveChats = () => {
  localStorage.setItem(chatsKey, JSON.stringify(chats));
};

const setActiveView = (viewName) => {
  views.forEach((view) => {
    const isActive = view.dataset.view === viewName;
    view.classList.toggle("active", isActive);
  });
};

const getCurrentChat = () => chats.find((c) => c.id === currentChatId);

const setCurrentChat = (chatId) => {
  currentChatId = chatId;
  localStorage.setItem(currentChatKey, chatId);
  senderId = `chat-${chatId}`;
};

const createChat = () => {
  const id = crypto.randomUUID();
  const chat = {
    id,
    title: `Chat ${chats.length + 1}`,
    createdAt: Date.now(),
    messages: [],
  };
  chats.push(chat);
  saveChats();
  setCurrentChat(id);
  renderMessages(chat);
  renderHistory();
  setActiveView("chat");
};

const renderMessages = (chat) => {
  messages.innerHTML = "";
  if (!chat) return;
  chat.messages.forEach((m) => addMessageToUI(m.text, m.role, m.ts));
  messages.scrollTop = messages.scrollHeight;
};

const addMessageToUI = (text, role = "user", ts = Date.now()) => {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  const meta = document.createElement("div");
  meta.className = "meta";
  const time = new Date(ts).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
  meta.textContent = `${role === "user" ? "You" : "Assistant"} - ${time}`;

  wrapper.appendChild(bubble);
  wrapper.appendChild(meta);
  messages.appendChild(wrapper);
  messages.scrollTop = messages.scrollHeight;
};

const addMessage = (text, role = "user") => {
  const chat = getCurrentChat();
  if (!chat) return;
  const entry = {
    id: crypto.randomUUID(),
    role,
    text,
    ts: Date.now(),
  };
  chat.messages.push(entry);
  saveChats();
  addMessageToUI(entry.text, entry.role, entry.ts);
  renderHistory();
};

const renderHistory = () => {
  if (!historyList) return;
  historyList.innerHTML = "";

  if (!chats.length) {
    const empty = document.createElement("div");
    empty.className = "history-empty";
    empty.textContent = "No chats yet.";
    historyList.appendChild(empty);
    return;
  }

  chats.slice().reverse().forEach((chat) => {
    const row = document.createElement("div");
    row.className = "history-item";
    row.dataset.chatId = chat.id;

    const label = document.createElement("div");
    label.className = "history-label";
    const last = chat.messages[chat.messages.length - 1];
    const preview = last ? `${last.role}: ${last.text}` : "(empty)";
    label.textContent = `${chat.title} - ${preview}`;

    const del = document.createElement("button");
    del.className = "delete";
    del.textContent = "Delete";
    del.addEventListener("click", (event) => {
      event.stopPropagation();
      chats = chats.filter((c) => c.id !== chat.id);
      if (currentChatId === chat.id) {
        currentChatId = null;
        if (chats.length) {
          setCurrentChat(chats[chats.length - 1].id);
          renderMessages(getCurrentChat());
        } else {
          messages.innerHTML = "";
        }
      }
      saveChats();
      renderHistory();
    });

    row.appendChild(label);
    row.appendChild(del);
    row.addEventListener("click", () => {
      setCurrentChat(chat.id);
      renderMessages(chat);
      setActiveView("chat");
      navItems.forEach((n) => n.classList.remove("active"));
      const chatTab = document.querySelector(".nav-item[data-view=\"chat\"]");
      if (chatTab) chatTab.classList.add("active");
    });

    historyList.appendChild(row);
  });
};

const setStatus = (text, connected) => {
  statusPill.textContent = text;
  statusPill.style.borderColor = connected
    ? "rgba(42, 215, 255, 0.6)"
    : "rgba(255, 189, 107, 0.6)";
  statusPill.style.color = connected ? "var(--accent-2)" : "#ffbd6b";
};

const sendToRasa = async (message) => {
  const payload = { sender: senderId, message };
  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }
  return res.json();
};

const sendMessage = async (text) => {
  addMessage(text, "user");
  sendButton.disabled = true;
  sendButton.textContent = "Sending...";

  try {
    const replies = await sendToRasa(text);
    setStatus("Connected", true);
    if (!replies.length) {
      addMessage("No reply from server. Check action server logs.", "assistant");
      return;
    }
    replies.forEach((reply) => {
      if (reply.text) {
        addMessage(reply.text, "assistant");
      } else if (reply.image) {
        addMessage(reply.image, "assistant");
      } else if (reply.custom) {
        addMessage(JSON.stringify(reply.custom), "assistant");
      }
    });
  } catch (err) {
    setStatus("Disconnected", false);
    addMessage("Connection failed. Check that Rasa server is running.", "assistant");
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = "Send";
  }
};

if (endpointLabel) {
  endpointLabel.textContent = endpoint;
}

newChatButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    createChat();
  });
});

if (clearHistoryButton) {
  clearHistoryButton.addEventListener("click", () => {
    chats = [];
    currentChatId = null;
    localStorage.removeItem(chatsKey);
    localStorage.removeItem(currentChatKey);
    renderHistory();
    messages.innerHTML = "";
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  await sendMessage(text);
});

quickButtons.forEach((btn) => {
  btn.addEventListener("click", (event) => {
    event.preventDefault();
    const text = btn.dataset.message || btn.textContent.trim();
    if (!text) return;
    sendMessage(text);
  });
});

navItems.forEach((item) => {
  item.addEventListener("click", () => {
    navItems.forEach((n) => n.classList.remove("active"));
    item.classList.add("active");
    const viewName = item.dataset.view;
    if (viewName) {
      setActiveView(viewName);
    }
  });
});

const checkStatus = async () => {
  try {
    const res = await fetch(`${baseUrl}/status`);
    setStatus(res.ok ? "Connected" : "Disconnected", res.ok);
  } catch {
    setStatus("Disconnected", false);
  }
};

const init = () => {
  loadChats();
  let savedId = localStorage.getItem(currentChatKey);
  if (!savedId || !chats.find((c) => c.id === savedId)) {
    if (chats.length) {
      savedId = chats[chats.length - 1].id;
    } else {
      createChat();
      return;
    }
  }
  setCurrentChat(savedId);
  renderMessages(getCurrentChat());
  renderHistory();
  setActiveView("chat");
  checkStatus();
};

init();
