'use client';

import React, { createContext, useContext, useState, useCallback } from 'react';

type NotificationType = 'success' | 'warning' | 'error' | 'info';

interface Notification {
  id: string;
  message: string;
  type: NotificationType;
  duration?: number;
}

interface NotificationContextType {
  notify: (message: string, type?: NotificationType, duration?: number) => void;
  confirm: (message: string) => Promise<boolean>;
  prompt: (message: string, defaultValue?: string) => Promise<string | null>;
}

const NotificationContext = createContext<NotificationContextType | null>(null);

export function NotificationProvider({ children }: { children: React.ReactNode }) {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [confirmDialog, setConfirmDialog] = useState<{
    message: string;
    resolve: (value: boolean) => void;
  } | null>(null);
  const [promptDialog, setPromptDialog] = useState<{
    message: string;
    defaultValue: string;
    resolve: (value: string | null) => void;
  } | null>(null);
  const [promptValue, setPromptValue] = useState('');

  const notify = useCallback((message: string, type: NotificationType = 'info', duration: number = 4000) => {
    const id = Date.now().toString();
    const notification = { id, message, type, duration };
    
    setNotifications(prev => [...prev, notification]);
    
    if (duration > 0) {
      setTimeout(() => {
        setNotifications(prev => prev.filter(n => n.id !== id));
      }, duration);
    }
  }, []);

  const confirm = useCallback((message: string): Promise<boolean> => {
    return new Promise((resolve) => {
      setConfirmDialog({ message, resolve });
    });
  }, []);

  const prompt = useCallback((message: string, defaultValue: string = ''): Promise<string | null> => {
    return new Promise((resolve) => {
      setPromptValue(defaultValue);
      setPromptDialog({ message, defaultValue, resolve });
    });
  }, []);

  const handleConfirm = (value: boolean) => {
    if (confirmDialog) {
      confirmDialog.resolve(value);
      setConfirmDialog(null);
    }
  };

  const handlePrompt = (value: string | null) => {
    if (promptDialog) {
      promptDialog.resolve(value);
      setPromptDialog(null);
      setPromptValue('');
    }
  };

  const getTypeStyles = (type: NotificationType) => {
    switch (type) {
      case 'success':
        return 'notification-success';
      case 'warning':
        return 'notification-warning';
      case 'error':
        return 'notification-error';
      default:
        return 'notification-info';
    }
  };

  return (
    <NotificationContext.Provider value={{ notify, confirm, prompt }}>
      {children}
      
      {/* Notification Stack */}
      <div className="notification-container">
        {notifications.map(notification => (
          <div
            key={notification.id}
            className={`notification ${getTypeStyles(notification.type)}`}
          >
            <div className="notification-content">
              {notification.message.split('\n').map((line, i) => (
                <div key={i}>{line}</div>
              ))}
            </div>
            <button
              onClick={() => setNotifications(prev => prev.filter(n => n.id !== notification.id))}
              className="notification-close"
            >
              ✕
            </button>
          </div>
        ))}
      </div>

      {/* Confirm Dialog */}
      {confirmDialog && (
        <>
          <div className="dialog-overlay" onClick={() => handleConfirm(false)} />
          <div className="dialog confirm-dialog">
            <div className="dialog-content">
              {confirmDialog.message.split('\n').map((line, i) => (
                <div key={i}>{line}</div>
              ))}
            </div>
            <div className="dialog-actions">
              <button onClick={() => handleConfirm(false)} className="dialog-button dialog-button-cancel">
                CANCEL
              </button>
              <button onClick={() => handleConfirm(true)} className="dialog-button dialog-button-confirm">
                CONFIRM
              </button>
            </div>
          </div>
        </>
      )}

      {/* Prompt Dialog */}
      {promptDialog && (
        <>
          <div className="dialog-overlay" onClick={() => handlePrompt(null)} />
          <div className="dialog prompt-dialog">
            <div className="dialog-content">
              {promptDialog.message.split('\n').map((line, i) => (
                <div key={i}>{line}</div>
              ))}
            </div>
            <input
              type="text"
              value={promptValue}
              onChange={(e) => setPromptValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handlePrompt(promptValue);
                if (e.key === 'Escape') handlePrompt(null);
              }}
              className="dialog-input"
              autoFocus
            />
            <div className="dialog-actions">
              <button onClick={() => handlePrompt(null)} className="dialog-button dialog-button-cancel">
                CANCEL
              </button>
              <button onClick={() => handlePrompt(promptValue)} className="dialog-button dialog-button-confirm">
                OK
              </button>
            </div>
          </div>
        </>
      )}

      <style jsx>{`
        /* Notification Styles - Paper Craft */
        .notification-container {
          position: fixed;
          top: 20px;
          right: 20px;
          z-index: 9999;
          display: flex;
          flex-direction: column;
          gap: 12px;
          pointer-events: none;
        }

        .notification {
          background: #FFFFFF;
          border: none;
          padding: 16px 20px;
          min-width: 320px;
          max-width: 420px;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          font-size: 14px;
          display: flex;
          justify-content: space-between;
          align-items: start;
          gap: 16px;
          pointer-events: auto;
          box-shadow: 2px 2px 0px rgba(0,0,0,0.1);
          border-radius: 12px;
          animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn {
          from {
            transform: translateX(100%);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }

        .notification-content {
          flex: 1;
          line-height: 1.5;
          font-weight: 400;
          color: #2C3E50;
        }

        .notification-close {
          background: none;
          border: none;
          padding: 4px;
          width: 24px;
          height: 24px;
          font-size: 18px;
          cursor: pointer;
          color: #7F8C8D;
          border-radius: 12px;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.2s;
        }

        .notification-close:hover {
          background: #F0EDE5;
          color: #2C3E50;
        }

        /* Paper craft colors with subtle backgrounds */
        .notification-success {
          background: #E8F8F5;
          color: #27AE60;
        }

        .notification-warning {
          background: #FEF5E7;
          color: #F39C12;
        }

        .notification-error {
          background: #FDEDEC;
          color: #E74C3C;
        }

        .notification-info {
          background: #EBF5FB;
          color: #3498DB;
        }

        /* Dialog Styles - Paper Craft */
        .dialog-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(44, 62, 80, 0.3);
          backdrop-filter: blur(2px);
          z-index: 9998;
        }

        .dialog {
          position: fixed;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          background: #FFFEF9;
          border: none;
          padding: 24px;
          min-width: 400px;
          max-width: 500px;
          z-index: 9999;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          font-size: 14px;
          box-shadow: 2px 2px 0px rgba(0,0,0,0.1);
          border-radius: 12px;
          animation: dialogFadeIn 0.3s ease-out;
        }

        @keyframes dialogFadeIn {
          from {
            opacity: 0;
            transform: translate(-50%, -48%);
          }
          to {
            opacity: 1;
            transform: translate(-50%, -50%);
          }
        }

        .dialog-content {
          margin-bottom: 24px;
          line-height: 1.6;
          font-weight: 400;
          color: #2C3E50;
        }

        .dialog-input {
          width: 100%;
          padding: 10px 12px;
          border: 1px solid #E1E8ED;
          background: #FFFFFF;
          font-family: inherit;
          font-size: inherit;
          margin-bottom: 20px;
          border-radius: 8px;
          transition: all 0.2s;
        }

        .dialog-input:focus {
          outline: none;
          border: 1px solid #3498DB;
          box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }

        .dialog-actions {
          display: flex;
          justify-content: flex-end;
          gap: 12px;
        }

        .dialog-button {
          padding: 10px 20px;
          border: none;
          background: #F8F7F3;
          font-family: inherit;
          font-size: 13px;
          font-weight: 500;
          cursor: pointer;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          border-radius: 8px;
          transition: all 0.2s;
          box-shadow: 1px 1px 0px rgba(0,0,0,0.1);
        }

        .dialog-button:hover {
          transform: translateY(-1px);
          box-shadow: 2px 2px 0px rgba(0,0,0,0.15);
        }

        .dialog-button-confirm {
          background: #3498DB;
          color: #FFFFFF;
        }

        .dialog-button-confirm:hover {
          background: #2980B9;
        }

        .dialog-button-cancel {
          background: #F0EDE5;
          color: #7F8C8D;
        }

        .dialog-button-cancel:hover {
          background: #E1E8ED;
          color: #2C3E50;
        }
      `}</style>
    </NotificationContext.Provider>
  );
}

export function useNotification() {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotification must be used within NotificationProvider');
  }
  return context;
}