import React, { useState, useEffect } from 'react';
import { X, Construction, ArrowRight } from 'lucide-react';
import './DevModal.css';

interface DevModalProps {
  onContinue?: () => void;
}

export default function DevModal({ onContinue }: DevModalProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [shouldShow, setShouldShow] = useState(false);

  useEffect(() => {
    // Check if user has seen the modal in this session
    const hasSeenModal = sessionStorage.getItem('effgen-dev-modal-seen');
    if (!hasSeenModal) {
      setShouldShow(true);
      // Small delay for animation
      setTimeout(() => setIsVisible(true), 100);
    }
  }, []);

  const handleClose = () => {
    setIsVisible(false);
    setTimeout(() => {
      setShouldShow(false);
      sessionStorage.setItem('effgen-dev-modal-seen', 'true');
      onContinue?.();
    }, 300);
  };

  if (!shouldShow) return null;

  return (
    <div className={`dev-modal-overlay ${isVisible ? 'visible' : ''}`} onClick={handleClose}>
      <div className={`dev-modal ${isVisible ? 'visible' : ''}`} onClick={e => e.stopPropagation()}>
        <button className="dev-modal-close" onClick={handleClose} aria-label="Close">
          <X size={20} />
        </button>

        <div className="dev-modal-icon">
          <Construction size={48} />
        </div>

        <h2 className="dev-modal-title">Documentation In Progress</h2>

        <p className="dev-modal-description">
          We're actively building comprehensive documentation for <strong>effGen</strong>.
          This page is currently under development.
        </p>

        <div className="dev-modal-progress">
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: '80%' }} />
          </div>
          <span className="progress-text">80% Complete</span>
        </div>

        <div className="dev-modal-features">
          <div className="feature-item completed">
            <span className="feature-check">&#10003;</span>
            <span>Core Documentation</span>
          </div>
          <div className="feature-item completed">
            <span className="feature-check">&#10003;</span>
            <span>API Reference</span>
          </div>
          <div className="feature-item completed">
            <span className="feature-check">&#10003;</span>
            <span>Examples & Guides</span>
          </div>
          <div className="feature-item">
            <span className="feature-pending">&#9711;</span>
            <span>Advanced Tutorials</span>
          </div>
        </div>

        <button className="dev-modal-button" onClick={handleClose}>
          <span>Continue to Docs</span>
          <ArrowRight size={18} />
        </button>
      </div>
    </div>
  );
}
