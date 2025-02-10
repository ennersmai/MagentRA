// frontend/src/stores/chat.js
import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useChatStore = defineStore('chat', () => {
  const messages = ref([])
  const sessionId = ref('global')

  // Function to send a message and retrieve the agent's response from the FastAPI backend.
  const sendMessage = async (payload) => {
    // Add the user's message to the store.
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: payload.text
    }
    messages.value.push(userMessage)

    // Call the backend /chat endpoint.
    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: payload.text,
          history: messages.value.map(m => m.content)
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
      messages.value.push(aiMessage)
    } catch (error) {
      console.error('Error fetching chat response:', error)
      const errorMessage = {
        id: Date.now() + 1,
        role: 'ai',
        content: 'Error: Unable to get response from server.'
      }
      messages.value.push(errorMessage)
    }
  }

  return { messages, sessionId, sendMessage }
})