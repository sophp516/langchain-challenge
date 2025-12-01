import { useState, useRef, useEffect } from 'react'
import { Client } from '@langchain/langgraph-sdk'
import { Send, Loader2, Plus } from "lucide-react"
import ReactMarkdown from 'react-markdown'
import { createThread as createThreadInDb, appendMessage } from '../services/db'
import type { AgentConfig } from './SettingsModal'
import { convertReferencesToLinks } from '../services/formatter'
import './ChatTab.css'


interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}


let client: Client | null = null
const getClient = () => {
  const apiUrl = import.meta.env.VITE_API_URL
  console.log('API URL:', apiUrl)
  if (!client) {
    client = new Client({
      apiUrl: apiUrl
    })
    console.log('Initialized LangGraph client with URL:', apiUrl)
  }
  return client
}

interface ChatTabProps {
  agentConfig: AgentConfig
}

function ChatTab({ agentConfig }: ChatTabProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [currentThreadId, setCurrentThreadId] = useState<string | undefined>(undefined)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const sendButtonRef = useRef<HTMLButtonElement>(null)
  const newThreadButtonRef = useRef<HTMLButtonElement>(null)
  const processedMessagesCountRef = useRef<number>(0) // Track how many backend messages we've processed

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Auto-resize textarea and match button height
  useEffect(() => {
    const textarea = textareaRef.current
    const sendButton = sendButtonRef.current
    const newThreadButton = newThreadButtonRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      const scrollHeight = textarea.scrollHeight
      const minHeight = 44 
      const maxHeight = 200
      const newHeight = input.trim() 
        ? Math.min(Math.max(scrollHeight, minHeight), maxHeight)
        : minHeight
      textarea.style.height = `${newHeight}px`
      
      // Match button heights to textarea height
      if (sendButton) {
        sendButton.style.height = `${newHeight}px`
      }
      if (newThreadButton) {
        newThreadButton.style.height = `${newHeight}px`
      }
    }
  }, [input])

  const handleSend = async () => {
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date()
    }

    const userInput = input.trim()
    setMessages(prev => [...prev, userMessage])
    setInput('')
    // Reset button heights
    if (sendButtonRef.current) {
      sendButtonRef.current.style.height = '44px'
    }
    if (newThreadButtonRef.current) {
      newThreadButtonRef.current.style.height = '44px'
    }
    setIsLoading(true)

    try {
      const client = getClient()

      // Create or reuse thread
      let threadId = currentThreadId
      if (!threadId) {
        const thread = await client.threads.create()
        threadId = thread.thread_id
        setCurrentThreadId(threadId)
        processedMessagesCountRef.current = 0 // Reset counter for new thread
        console.log('Created new thread:', threadId)

        // Store thread in MongoDB
        try {
          await createThreadInDb(threadId)
        } catch (dbError) {
          console.error('Failed to store thread in MongoDB:', dbError)
        }
      }

      // Store user message in MongoDB
      try {
        await appendMessage(threadId, userMessage)
      } catch (dbError) {
        console.error('Failed to store user message in MongoDB:', dbError)
      }

      // Check if we need to resume from interrupt
      const state = await client.threads.getState(threadId)
      const isInterrupted = state.next?.includes('collect_response') || state.next?.includes('collect_feedback')

      let streamIterator
      if (isInterrupted) {
        // Resume from interrupt
        console.log('Resuming from interrupt')
        streamIterator = client.runs.stream(
          threadId,
          'agent',
          {
            command: { resume: userInput },
            streamMode: ['values', 'updates', 'messages'],
            config: { configurable: agentConfig }
          } as any
        )
      } else {
        // Start new run
        const input = {
          topic: 'userInput',
          messages: [{ type: 'human', content: userInput }],
          is_finalized: false,
          clarification_rounds: 0,
          clarification_questions: [],
          user_responses: [],
          subtopics: [],
          sub_researchers: [],
          report_outline: {},
          report_sections: [],
          report_history: [],
          current_report_id: 0,
          report_content: '',
          report_summary: '',
          report_conclusion: '',
          report_recommendations: [],
          report_references: [],
          report_citations: [],
          report_footnotes: [],
          report_endnotes: [],
        }

        streamIterator = client.runs.stream(
          threadId,
          'agent',
          {
            input,
            streamMode: ['values', 'updates', 'messages'],
            config: { configurable: agentConfig }
          } as any
        )
      }

      // Process stream
      for await (const chunk of streamIterator) {
        console.log('[LangGraph SDK] Received chunk:', chunk.event, new Date().toISOString())

        if (chunk.event === 'values' && chunk.data) {
          // Extract messages from state
          const state = chunk.data as any // Type assertion for LangGraph state
          if (state && typeof state === 'object' && state.messages && Array.isArray(state.messages)) {
            const backendMessages = state.messages
            const alreadyProcessed = processedMessagesCountRef.current

            // Filter for AI messages only (exclude human, user, and tool messages)
            const aiMessages = backendMessages
              .filter((msg: any) => {
                if (typeof msg === 'string') return true
                if (!msg || typeof msg !== 'object') return false
                const msgType = msg.type || msg.role || ''
                // Exclude human/user messages AND tool messages (raw JSON responses)
                return msgType !== 'human' && msgType !== 'user' && msgType !== 'tool'
              })

            // Only process new messages we haven't seen yet
            if (aiMessages.length > alreadyProcessed) {
              const newMessages = aiMessages.slice(alreadyProcessed)

              // Convert backend messages to UI messages
              const newUIMessages: Message[] = newMessages.map((msg: any, idx: number) => {
                let content = ''
                if (typeof msg === 'string') {
                  content = msg
                } else if (msg && typeof msg === 'object' && 'content' in msg) {
                  content = msg.content || ''
                }

                return {
                  id: `backend-${Date.now()}-${alreadyProcessed + idx}`,
                  role: 'assistant' as const,
                  content,
                  timestamp: new Date()
                }
              }).filter((msg: Message) => msg.content.trim() !== '')

              if (newUIMessages.length > 0) {
                console.log('[LangGraph SDK] Adding', newUIMessages.length, 'new messages')
                setMessages(prev => [...prev, ...newUIMessages])
                processedMessagesCountRef.current = aiMessages.length
                scrollToBottom()

                // Store assistant messages in MongoDB
                for (const msg of newUIMessages) {
                  try {
                    await appendMessage(threadId, msg)
                  } catch (dbError) {
                    console.error('Failed to store assistant message in MongoDB:', dbError)
                  }
                }
              }
            }
          }
        } else if (chunk.event === 'updates' && chunk.data) {
          // Handle incremental updates
          const updateData = chunk.data as any
          console.log('[LangGraph SDK] Got updates:', updateData)
        }
      }

      console.log('Stream completed')
    } catch (error) {
      console.error('Error during streaming:', error)
      // Add error message
      setMessages(prev => [...prev, {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date()
      }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleNewThread = () => {
    setMessages([])
    setCurrentThreadId(undefined)
    processedMessagesCountRef.current = 0
    setInput('')
    // Reset button heights
    if (sendButtonRef.current) {
      sendButtonRef.current.style.height = '44px'
    }
    if (newThreadButtonRef.current) {
      newThreadButtonRef.current.style.height = '44px'
    }
    if (textareaRef.current) {
      textareaRef.current.style.height = '44px'
    }
  }

  return (
    <div className="chat-tab">
      <div className="chat-messages">
        {messages.map((message) => (
          <div key={message.id} className={`message ${message.role}`}>
            <div className="message-header">
              <span className="message-role">{message.role === 'user' ? 'You' : 'Agent'}</span>
            </div>
            <div className="message-content">
              {message.content ? (
                <ReactMarkdown
                  components={{
                    a: ({ node, ...props }) => (
                      <a {...props} target="_blank" rel="noopener noreferrer" />
                    ),
                  }}
                >
                  {convertReferencesToLinks(message.content)}
                </ReactMarkdown>
              ) : (
                <div className="loading">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && messages.filter(m => m.role === 'assistant' && m.content === '').length === 0 && (
          <div className="message assistant">
            <div className="message-header">
              <span className="message-role">Agent</span>
            </div>
            <div className="message-content loading">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-container">
        <button
          ref={newThreadButtonRef}
          className="new-thread-button"
          onClick={handleNewThread}
          disabled={isLoading}
          title="Start New Thread"
        >
          <Plus size={20} />
        </button>
        <textarea
          ref={textareaRef}
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything"
          rows={1}
          disabled={isLoading}
        />
        <button
          ref={sendButtonRef}
          className="send-button"
          onClick={handleSend}
          disabled={isLoading || !input.trim()}
        >
          {isLoading ? <Loader2 className="send-button-icon" /> : <Send className="send-button-icon" />}
        </button>
      </div>
    </div>
  )
}

export default ChatTab

