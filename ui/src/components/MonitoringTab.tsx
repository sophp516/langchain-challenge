import { useState, useEffect } from 'react'
import { getAllThreads, fetchThreadMessages, deleteThread } from '../services/db'
import type { ThreadData, Message as ThreadMessage } from '../services/db'
import { Loader2, Trash2 } from 'lucide-react'
import { convertReferencesToLinks } from '../services/formatter'
import ReactMarkdown from 'react-markdown'
import './MonitoringTab.css'


function MonitoringTab() {
  const [threads, setThreads] = useState<ThreadData[]>([])
  const [selectedThreadId, setSelectedThreadId] = useState<string | undefined>(undefined)
  const [isLoading, setIsLoading] = useState(true)
  const [isLoadingMessages, setIsLoadingMessages] = useState(false)
  const [messages, setMessages] = useState<ThreadMessage[]>([])

  useEffect(() => {
    const fetchThreads = async () => {
      setIsLoading(true)
      try {
        const threads = await getAllThreads()
        setThreads(threads)
      } catch (error) {
        console.error('Failed to fetch threads:', error)
      } finally {
        setIsLoading(false)
      }
    }
    fetchThreads()
  }, [])

  useEffect(() => {
    if (selectedThreadId) {
      const fetchMessages = async () => {
        setIsLoadingMessages(true)
        try {
          const messages = await fetchThreadMessages(selectedThreadId)
          setMessages(messages)
        } catch (error) {
          console.error('Failed to fetch messages:', error)
        } finally {
          setIsLoadingMessages(false)
        }
      }
      fetchMessages()
    } else {
      setMessages([])
    }
  }, [selectedThreadId])

  const formatDate = (date: Date | string) => {
    const d = typeof date === 'string' ? new Date(date) : date
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(d)
  }

  const handleDeleteThread = async (threadId: string, e: React.MouseEvent) => {
    e.stopPropagation() // Prevent thread selection when clicking delete
    
    if (!window.confirm(`Are you sure you want to delete thread ${threadId}? This action cannot be undone.`)) {
      return
    }

    try {
      await deleteThread(threadId)
      
      // Remove from local state
      setThreads(prev => prev.filter(t => t.thread_id !== threadId))
      
      // If deleted thread was selected, clear selection
      if (selectedThreadId === threadId) {
        setSelectedThreadId(undefined)
        setMessages([])
      }
    } catch (error) {
      console.error('Failed to delete thread:', error)
      alert('Failed to delete thread. Please try again.')
    }
  }

  return (
    <div className="monitoring-tab">
      <div className="monitoring-header">
        <h2>Thread Monitoring</h2>
        <p className="monitoring-subtitle">
          {threads.length} {threads.length === 1 ? 'thread' : 'threads'} total
        </p>
      </div>
      <div className="monitoring-content">
        {isLoading ? (
          <div className="loading-state">
            <Loader2 className="loading-spinner" />
            <p>Loading threads...</p>
          </div>
        ) : threads.length === 0 ? (
          <div className="empty-state">
            <p>No threads found</p>
            <p className="empty-state-subtitle">Start a conversation in the Chat tab to see threads here</p>
          </div>
        ) : (
          <div className="monitoring-layout">
            <div className="threads-panel">
              <h3 className="panel-title">Threads</h3>
              <div className="threads-list">
                {threads.map((thread) => (
                  <div
                    key={thread.thread_id}
                    className={`thread-item ${selectedThreadId === thread.thread_id ? 'selected' : ''}`}
                    onClick={() => setSelectedThreadId(thread.thread_id)}
                  >
                    <div className="thread-content">
                      <div className="thread-id">{thread.thread_id}</div>
                      <div className="thread-meta">
                        <span className="thread-date">{formatDate(thread.created_at)}</span>
                        <span className="thread-count">{thread.messages?.length || 0} messages</span>
                      </div>
                    </div>
                    <button
                      className="thread-delete-button"
                      onClick={(e) => handleDeleteThread(thread.thread_id, e)}
                      title="Delete thread"
                      aria-label="Delete thread"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
            <div className="messages-panel">
              {isLoadingMessages ? (
                <div className="loading-state">
                  <Loader2 className="loading-spinner" />
                  <p>Loading messages...</p>
                </div>
              ) : selectedThreadId ? (
                messages.length === 0 ? (
                  <div className="empty-state">
                    <p>No messages in this thread</p>
                  </div>
                ) : (
                  <div className="messages-list">
                    {messages.map((message) => (
                      <div key={message.id} className={`message-item ${message.role}`}>
                        <div className="message-header">
                          <span className="message-role">{message.role === 'user' ? 'User' : 'Agent'}</span>
                        </div>
                        <div className="message-content">
                          <ReactMarkdown
                            components={{
                              a: ({ node, ...props }) => (
                                <a {...props} target="_blank" rel="noopener noreferrer" />
                              ),
                            }}
                          >
                            {convertReferencesToLinks(message.content)}
                          </ReactMarkdown>
                        </div>
                      </div>
                    ))}
                  </div>
                )
              ) : (
                <div className="empty-state">
                  <p>Select a thread to view messages</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default MonitoringTab

