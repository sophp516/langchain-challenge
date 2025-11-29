import { useState, useEffect } from 'react'
import { X } from 'lucide-react'
import './SettingsModal.css'

export interface AgentConfig {
  search_api: 'tavily'
  max_search_results: number
  max_research_depth: number
  max_subtopics: number
  max_clarification_rounds: number
  max_revision_rounds: number
  min_credibility_score: number
  // UI-only fields (not in backend config)
  min_report_score?: number
  enable_user_feedback?: boolean
  enable_cross_verification?: boolean
}

const defaultConfig: AgentConfig = {
  search_api: 'tavily',
  max_search_results: 5,
  max_research_depth: 3,
  max_subtopics: 7,
  max_clarification_rounds: 0,
  max_revision_rounds: 1,
  min_credibility_score: 0.3,
  // UI-only defaults
  min_report_score: 85,
  enable_user_feedback: true,
  enable_cross_verification: false
}

interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
  onSave: (config: AgentConfig) => void
  currentConfig: AgentConfig
}

export function SettingsModal({ isOpen, onClose, onSave, currentConfig }: SettingsModalProps) {
  const [config, setConfig] = useState<AgentConfig>(currentConfig)

  useEffect(() => {
    if (isOpen) {
      setConfig(currentConfig)
    }
  }, [isOpen, currentConfig])

  const handleSave = () => {
    onSave(config)
    onClose()
  }

  const handleReset = () => {
    setConfig(defaultConfig)
  }

  if (!isOpen) return null

  return (
    <div className="settings-modal-overlay" onClick={onClose}>
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <div className="settings-modal-header">
          <h2>Agent Configuration</h2>
          <button className="settings-modal-close" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        <div className="settings-modal-content">
          {/* Search API Settings */}
          <div className="settings-section">
            <h3>Search API Settings</h3>
            <div className="settings-field">
              <label>Search API</label>
              <select
                value={config.search_api}
                style={{ opacity: 0.7 }}
              >
                <option value="tavily">Tavily</option>
                <option value="exa">Exa</option>
              </select>
            </div>
            <div className="settings-field">
              <label>Max Search Results (1-20)</label>
              <input
                type="number"
                min="1"
                max="20"
                value={config.max_search_results}
                onChange={(e) => setConfig({ ...config, max_search_results: parseInt(e.target.value) || 5 })}
              />
            </div>
          </div>

          {/* Research Depth Settings */}
          <div className="settings-section">
            <h3>Research Depth Settings</h3>
            <div className="settings-field">
              <label>Max Research Depth (1-5)</label>
              <input
                type="number"
                min="1"
                max="5"
                value={config.max_research_depth}
                onChange={(e) => setConfig({ ...config, max_research_depth: parseInt(e.target.value) || 3 })}
              />
            </div>
            <div className="settings-field">
              <label>Max Subtopics (3-10)</label>
              <input
                type="number"
                min="3"
                max="10"
                value={config.max_subtopics}
                onChange={(e) => setConfig({ ...config, max_subtopics: parseInt(e.target.value) || 7 })}
              />
            </div>
          </div>

          {/* Clarification Settings */}
          <div className="settings-section">
            <h3>Clarification Settings</h3>
            <div className="settings-field">
              <label>Max Clarification Rounds (0-10)</label>
              <input
                type="number"
                min="0"
                max="10"
                value={config.max_clarification_rounds}
                onChange={(e) => setConfig({ ...config, max_clarification_rounds: parseInt(e.target.value) || 0 })}
              />
            </div>
          </div>

          {/* Report Quality Settings */}
          <div className="settings-section">
            <h3>Report Quality Settings</h3>
            <div className="settings-field">
              <label>Min Report Score (0-100)</label>
              <input
                type="number"
                min="0"
                max="100"
                value={config.min_report_score}
                onChange={(e) => setConfig({ ...config, min_report_score: parseInt(e.target.value) || 85 })}
              />
            </div>
            <div className="settings-field">
              <label>Max Revision Rounds (0-5)</label>
              <input
                type="number"
                min="0"
                max="5"
                value={config.max_revision_rounds}
                onChange={(e) => setConfig({ ...config, max_revision_rounds: parseInt(e.target.value) || 1 })}
              />
            </div>
          </div>

          {/* Feature Toggles */}
          <div className="settings-section">
            <h3>Feature Toggles</h3>
            <div className="settings-field checkbox-field">
              <label>
                <input
                  type="checkbox"
                  checked={config.enable_user_feedback}
                  onChange={(e) => setConfig({ ...config, enable_user_feedback: e.target.checked })}
                />
                Enable User Feedback
              </label>
            </div>
            <div className="settings-field checkbox-field">
              <label>
                <input
                  type="checkbox"
                  checked={config.enable_cross_verification}
                  onChange={(e) => setConfig({ ...config, enable_cross_verification: e.target.checked })}
                />
                Enable Cross-Reference Verification
              </label>
            </div>
          </div>

          {/* Source Quality Settings */}
          <div className="settings-section">
            <h3>Source Quality Settings</h3>
            <div className="settings-field">
              <label>Min Credibility Score (0.0-1.0)</label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.1"
                value={config.min_credibility_score}
                onChange={(e) => setConfig({ ...config, min_credibility_score: parseFloat(e.target.value) || 0.3 })}
              />
            </div>
          </div>
        </div>

        <div className="settings-modal-footer">
          <button className="settings-button-secondary" onClick={handleReset}>
            Reset to Defaults
          </button>
          <div className="settings-button-group">
            <button className="settings-button-secondary" onClick={onClose}>
              Cancel
            </button>
            <button className="settings-button-primary" onClick={handleSave}>
              Save
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

