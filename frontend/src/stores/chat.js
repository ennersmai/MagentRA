// frontend/src/stores/chat.js
import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useChatStore = defineStore('chat', () => {
  // Initialize history as a reactive array
  const history = ref([])

  // Optionally, add other properties, e.g. sessionId
  const sessionId = ref(Date.now().toString())

  // Function to send a message and retrieve the agent's response from the FastAPI backend.
  const sendMessage = async (payload) => {
    // Append messages using store.history.value.push(...)
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: payload.text
    }
    history.value.push(userMessage)

    // Call the backend /chat endpoint.
    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: payload.text,
          history: history.value.map(m => m.content)
        })
      })

      if (!response.ok) {
        throw new Error('Network response was not ok')
      }
      
      const data = await response.json()
      
      // Append the AI response.
      const aiMessage = {
        id: Date.now() + 1,
        role: 'ai',
        content: data.response
      }
      history.value.push(aiMessage)
    } catch (error) {
      console.error('Error fetching chat response:', error)
      const errorMessage = {
        id: Date.now() + 1,
        role: 'ai',
        content: 'Error: Unable to get response from server.'
      }
      history.value.push(errorMessage)
    }
  }

  return { history, sessionId, sendMessage }
})