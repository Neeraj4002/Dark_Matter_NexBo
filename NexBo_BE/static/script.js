document.addEventListener("DOMContentLoaded", function () {
    const sendBtn = document.getElementById("send-btn");
    const userInput = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
  
    sendBtn.addEventListener("click", function () {
      const message = userInput.value.trim();
      if (message === "") return;
  
      appendMessage("user", message);
      userInput.value = "";
      
      // Call the /chat endpoint.
      fetch("/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: message })
      })
      .then(response => response.json())
      .then(data => {
        if (data.response) {
          appendMessage("assistant", data.response);
        }
      })
      .catch(error => console.error("Error:", error));
    });
  
    // Optionally, allow sending message on pressing Enter.
    userInput.addEventListener("keypress", function (e) {
      if (e.key === "Enter") {
        sendBtn.click();
      }
    });
  
    function appendMessage(role, text) {
      const messageDiv = document.createElement("div");
      messageDiv.classList.add("message", role);
      messageDiv.innerText = text;
      chatBox.appendChild(messageDiv);
      chatBox.scrollTop = chatBox.scrollHeight;
    }
  });
  