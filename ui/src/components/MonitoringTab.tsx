import { useState, useEffect } from 'react'
import { getAllThreads, fetchThreadMessages } from '../services/db'
import type { ThreadData, Message as ThreadMessage } from '../services/db'

function MonitoringTab() {

  const [threads, setThreads] = useState<ThreadData[]>([])
  const [selectedThreadId, setSelectedThreadId] = useState<string | undefined>(undefined)
  const [isLoading, setIsLoading] = useState(true)
  const [messages, setMessages] = useState<ThreadMessage[]>([])

  useEffect(() => {
    const fetchThreads = async () => {
      setIsLoading(true)
      const threads = await getAllThreads()
      setThreads(threads)
      setIsLoading(false)
    }
    fetchThreads()
  }, [])

  useEffect(() => {
    if (selectedThreadId) {
      const fetchMessages = async () => {
        const messages = await fetchThreadMessages(selectedThreadId)
        setMessages(messages)
      }
      fetchMessages()
    }
  }, [selectedThreadId])

  return (
    <div className="monitoring-tab">
      <div className="monitoring-content">
        <div className="empty-state">
          {isLoading ? (
            <p>Loading threads...</p>
          ) : (
            <div>
            <div className="threads-list">
              {threads.map((thread) => (
                <div key={thread.thread_id} className="thread-item" onClick={() => setSelectedThreadId(thread.thread_id)}>
                  {thread.thread_id}
                </div>
              ))}
            </div>
            <div className="messages-list">
              {messages.map((message) => (
                <div key={message.id} className="message-item">
                  {message.content}
                </div>
              ))}
            </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default MonitoringTab

