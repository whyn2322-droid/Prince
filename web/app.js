const messages = document.getElementById("messages");
const composer = document.getElementById("composer");
const messageInput = document.getElementById("messageInput");
const serverUrlInput = document.getElementById("serverUrl");
const saveServerBtn = document.getElementById("saveServer");
const statusText = document.getElementById("statusText");
const quickReplies = document.getElementById("quickReplies");

const DEFAULT_SERVER = "http://localhost:5005";
const STORAGE_KEY = "rasa_server_url";

const state = {
  serverUrl: localStorage.getItem(STORAGE_KEY) || DEFAULT_SERVER,
  senderId: "web_user",
};

serverUrlInput.value = state.serverUrl;

function setStatus(connected) {
  statusText.textContent = connected ? "Холбогдсон" : "Салсан";
  statusText.previousElementSibling.style.background = connected
    ? "#22c55e"
    : "#f59e0b";
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;");
}

function addMessage({ text, type = "incoming", imageUrl, meta }) {
  const bubble = document.createElement("div");
  bubble.className = `message ${type}`;

  if (text) {
    const p = document.createElement("p");
    p.innerHTML = escapeHtml(text).replace(/\n/g, "<br />");
    bubble.appendChild(p);
  }

  if (imageUrl) {
    const img = document.createElement("img");
    img.src = imageUrl;
    img.alt = "bot illustration";
    bubble.appendChild(img);
  }

  if (meta) {
    const metaEl = document.createElement("div");
    metaEl.className = "meta";
    metaEl.textContent = meta;
    bubble.appendChild(metaEl);
  }

  messages.appendChild(bubble);
  messages.scrollTop = messages.scrollHeight;
}

function addTypingIndicator() {
  const bubble = document.createElement("div");
  bubble.className = "message incoming";
  bubble.id = "typing";

  const typing = document.createElement("div");
  typing.className = "typing";
  typing.innerHTML = "<span></span><span></span><span></span>";
  bubble.appendChild(typing);

  messages.appendChild(bubble);
  messages.scrollTop = messages.scrollHeight;
}

function removeTypingIndicator() {
  const typing = document.getElementById("typing");
  if (typing) typing.remove();
}

async function sendMessage(text) {
  if (!text.trim()) return;

  addMessage({
    text,
    type: "outgoing",
    meta: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
  });

  addTypingIndicator();

  try {
    const response = await fetch(`${state.serverUrl}/webhooks/rest/webhook`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sender: state.senderId, message: text }),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();
    setStatus(true);

    if (!Array.isArray(data) || data.length === 0) {
      addMessage({
        text: "Хариу ирсэнгүй. Өөрөөр асуугаад үзээрэй.",
        type: "incoming",
      });
      return;
    }

    data.forEach((item) => {
      if (item.text) {
        addMessage({
          text: item.text,
          type: "incoming",
          meta: "MathBot",
        });
      }

      if (item.image) {
        addMessage({
          text: item.text || "",
          type: "incoming",
          imageUrl: item.image,
          meta: "MathBot",
        });
      }

      if (Array.isArray(item.buttons)) {
        item.buttons.forEach((btn) => {
          const button = document.createElement("button");
          button.className = "chip";
          button.textContent = btn.title || btn.payload;
          button.addEventListener("click", () => {
            sendMessage(btn.payload || btn.title);
          });
          quickReplies.appendChild(button);
        });
      }
    });
  } catch (error) {
    setStatus(false);
    addMessage({
      text:
        "Сервертэй холбогдож чадсангүй. Rasa ажиллаж байгаа эсэх, мөн CORS нээлттэй эсэхийг шалгана уу.",
      type: "incoming",
    });
  } finally {
    removeTypingIndicator();
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = messageInput.value;
  messageInput.value = "";
  sendMessage(text);
});

quickReplies.addEventListener("click", (event) => {
  const chip = event.target.closest("button");
  if (chip) {
    sendMessage(chip.dataset.text || chip.textContent);
  }
});

saveServerBtn.addEventListener("click", () => {
  const newUrl = serverUrlInput.value.trim() || DEFAULT_SERVER;
  state.serverUrl = newUrl.replace(/\/$/, "");
  localStorage.setItem(STORAGE_KEY, state.serverUrl);
  setStatus(false);
});

window.addEventListener("load", () => {
  addMessage({
    text: "Сайн уу! MathBot-д тавтай морил. Математикийн асуулт асуугаарай.",
    type: "incoming",
    meta: "MathBot",
  });
});
