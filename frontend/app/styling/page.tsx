'use client';

import { useState } from 'react';
import { useNotification } from '@/components/NotificationProvider';

type StyleProposal = 'current' | 'soft' | 'neo' | 'paper';

export default function StylingPage() {
  const [activeProposal, setActiveProposal] = useState<StyleProposal>('current');
  const { notify } = useNotification();

  const proposals = {
    current: {
      name: 'Current (Brutalist)',
      description: 'Sharp corners, thick borders, minimal labels',
    },
    soft: {
      name: 'Soft Minimal',
      description: 'Clean, accessible design with subtle borders and clear labeling',
    },
    neo: {
      name: 'Neo-Terminal',
      description: 'Modern terminal aesthetic with dotted borders and green accents',
    },
    paper: {
      name: 'Paper Craft',
      description: 'Paper-inspired with soft shadows and warm tones',
    },
  };

  const applyStyle = (style: StyleProposal) => {
    // In a real implementation, this would switch CSS files
    notify(`Style "${proposals[style].name}" would be applied globally`, 'info');
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold mb-2">DESIGN SYSTEM PROPOSALS</h1>
        <p className="text-gray-600">
          Choose a design system that balances aesthetics with usability
        </p>
      </div>

      {/* Proposal Selector */}
      <div className="flex gap-4 border-b border-black pb-4">
        {Object.entries(proposals).map(([key, proposal]) => (
          <button
            key={key}
            onClick={() => setActiveProposal(key as StyleProposal)}
            className={`px-4 py-2 ${
              activeProposal === key
                ? 'bg-black text-white'
                : 'border border-gray-300 hover:bg-gray-50'
            }`}
          >
            {proposal.name}
          </button>
        ))}
      </div>

      {/* Current Brutalist Style */}
      {activeProposal === 'current' && (
        <div className="space-y-6">
          <div className="border-2 border-black p-6">
            <h2 className="text-lg font-bold mb-4">CURRENT BRUTALIST STYLE</h2>
            
            {/* Sample Table */}
            <div className="mb-6">
              <h3 className="font-bold mb-2">TABLE DESIGN</h3>
              <table className="w-full border-2 border-black">
                <thead>
                  <tr className="border-b-2 border-black">
                    <th className="text-left p-2">ID</th>
                    <th className="text-left p-2">TITLE</th>
                    <th className="text-left p-2">STATUS</th>
                    <th className="text-left p-2">ACTIONS</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-black">
                    <td className="p-2">001</td>
                    <td className="p-2">Sample Document</td>
                    <td className="p-2 text-green-600">ACTIVE</td>
                    <td className="p-2 space-x-2">
                      <button className="text-blue-600">[E]</button>
                      <button className="text-red-600">[D]</button>
                      <button className="text-green-600">[R]</button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Buttons */}
            <div className="mb-6">
              <h3 className="font-bold mb-2">BUTTONS</h3>
              <div className="space-x-2">
                <button className="px-4 py-2 border-2 border-black">PRIMARY</button>
                <button className="px-4 py-2 bg-black text-white">SECONDARY</button>
                <button className="px-4 py-2 border-2 border-gray-400 text-gray-600" disabled>DISABLED</button>
              </div>
            </div>

            {/* Form Elements */}
            <div className="mb-6">
              <h3 className="font-bold mb-2">FORM ELEMENTS</h3>
              <div className="space-y-2">
                <input type="text" placeholder="Text input" className="w-full border border-black p-2" />
                <select className="w-full border border-black p-2">
                  <option>Select option</option>
                  <option>Option 1</option>
                  <option>Option 2</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Soft Minimal Style */}
      {activeProposal === 'soft' && (
        <div className="space-y-6">
          <div style={{ border: '1px solid #E5E7EB', padding: '24px', borderRadius: '8px' }}>
            <h2 className="text-lg font-bold mb-4">SOFT MINIMAL STYLE</h2>
            
            {/* Sample Table */}
            <div className="mb-6">
              <h3 className="font-semibold mb-3 text-gray-700">Table Design</h3>
              <table className="w-full" style={{ border: '1px solid #E5E7EB', borderRadius: '4px', overflow: 'hidden' }}>
                <thead style={{ backgroundColor: '#F9FAFB' }}>
                  <tr style={{ borderBottom: '1px solid #E5E7EB' }}>
                    <th className="text-left p-3 font-medium">ID</th>
                    <th className="text-left p-3 font-medium">Title</th>
                    <th className="text-left p-3 font-medium">Status</th>
                    <th className="text-left p-3 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: '1px solid #F3F4F6' }}>
                    <td className="p-3 text-gray-600">001</td>
                    <td className="p-3">Sample Document</td>
                    <td className="p-3">
                      <span style={{ color: '#10B981', backgroundColor: '#ECFDF5', padding: '2px 8px', borderRadius: '4px', fontSize: '12px' }}>
                        Active
                      </span>
                    </td>
                    <td className="p-3 space-x-2">
                      <button style={{ color: '#3B82F6', padding: '4px 8px', borderRadius: '4px', border: '1px solid #DBEAFE' }}>
                        Edit
                      </button>
                      <button style={{ color: '#EF4444', padding: '4px 8px', borderRadius: '4px', border: '1px solid #FEE2E2' }}>
                        Delete
                      </button>
                      <button style={{ color: '#10B981', padding: '4px 8px', borderRadius: '4px', border: '1px solid #D1FAE5' }}>
                        Rechunk
                      </button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Buttons */}
            <div className="mb-6">
              <h3 className="font-semibold mb-3 text-gray-700">Buttons</h3>
              <div className="space-x-3">
                <button style={{ 
                  backgroundColor: '#1A1A1A', 
                  color: 'white', 
                  padding: '8px 16px', 
                  borderRadius: '4px',
                  border: 'none'
                }}>
                  Primary Button
                </button>
                <button style={{ 
                  backgroundColor: 'transparent', 
                  color: '#1A1A1A', 
                  padding: '8px 16px', 
                  borderRadius: '4px',
                  border: '1px solid #E5E7EB'
                }}>
                  Secondary Button
                </button>
                <button style={{ 
                  backgroundColor: '#F9FAFB', 
                  color: '#9CA3AF', 
                  padding: '8px 16px', 
                  borderRadius: '4px',
                  border: '1px solid #E5E7EB',
                  cursor: 'not-allowed'
                }} disabled>
                  Disabled
                </button>
              </div>
            </div>

            {/* Form Elements */}
            <div className="mb-6">
              <h3 className="font-semibold mb-3 text-gray-700">Form Elements</h3>
              <div className="space-y-3">
                <input 
                  type="text" 
                  placeholder="Enter text here..." 
                  style={{ 
                    width: '100%', 
                    border: '1px solid #E5E7EB', 
                    padding: '8px 12px',
                    borderRadius: '4px',
                    backgroundColor: 'white'
                  }} 
                />
                <select style={{ 
                  width: '100%', 
                  border: '1px solid #E5E7EB', 
                  padding: '8px 12px',
                  borderRadius: '4px',
                  backgroundColor: 'white'
                }}>
                  <option>Select an option...</option>
                  <option>Option 1</option>
                  <option>Option 2</option>
                </select>
              </div>
            </div>

            <button 
              onClick={() => applyStyle('soft')}
              style={{ 
                backgroundColor: '#0066FF', 
                color: 'white', 
                padding: '10px 24px', 
                borderRadius: '4px',
                border: 'none',
                fontWeight: '500'
              }}
            >
              Use Soft Minimal Style
            </button>
          </div>
        </div>
      )}

      {/* Neo-Terminal Style */}
      {activeProposal === 'neo' && (
        <div className="space-y-6">
          <div style={{ 
            border: '1px dashed #00AA00', 
            padding: '24px', 
            borderRadius: '2px',
            backgroundColor: '#0A0A0A',
            color: '#00FF00'
          }}>
            <h2 className="text-lg font-bold mb-4" style={{ color: '#00FF00' }}>
              {'>'} NEO-TERMINAL STYLE
            </h2>
            
            {/* Sample Table */}
            <div className="mb-6">
              <h3 className="font-bold mb-2" style={{ color: '#00AA00' }}>
                :: TABLE_DESIGN ::
              </h3>
              <table className="w-full" style={{ borderCollapse: 'separate', borderSpacing: '0' }}>
                <thead>
                  <tr style={{ borderBottom: '1px dotted #00AA00' }}>
                    <th className="text-left p-2" style={{ color: '#00FF00' }}>ID</th>
                    <th className="text-left p-2" style={{ color: '#00FF00' }}>TITLE</th>
                    <th className="text-left p-2" style={{ color: '#00FF00' }}>STATUS</th>
                    <th className="text-left p-2" style={{ color: '#00FF00' }}>ACTIONS</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: '1px dotted #006600' }}>
                    <td className="p-2" style={{ color: '#00AA00' }}>001</td>
                    <td className="p-2" style={{ color: '#00FF00' }}>Sample_Document</td>
                    <td className="p-2">
                      <span style={{ color: '#00FF00' }}>[ACTIVE]</span>
                    </td>
                    <td className="p-2 space-x-2">
                      <button style={{ color: '#00AAFF', backgroundColor: 'transparent', border: 'none' }}>
                        📝 Edit
                      </button>
                      <button style={{ color: '#FF0000', backgroundColor: 'transparent', border: 'none' }}>
                        🗑 Del
                      </button>
                      <button style={{ color: '#00FF00', backgroundColor: 'transparent', border: 'none' }}>
                        ↻ Rech
                      </button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Buttons */}
            <div className="mb-6">
              <h3 className="font-bold mb-2" style={{ color: '#00AA00' }}>
                :: BUTTONS ::
              </h3>
              <div className="space-x-3">
                <button style={{ 
                  backgroundColor: '#00AA00', 
                  color: '#000000', 
                  padding: '8px 16px', 
                  borderRadius: '2px',
                  border: 'none',
                  fontFamily: 'monospace'
                }}>
                  [PRIMARY]
                </button>
                <button style={{ 
                  backgroundColor: 'transparent', 
                  color: '#00FF00', 
                  padding: '8px 16px', 
                  borderRadius: '2px',
                  border: '1px solid #00AA00',
                  fontFamily: 'monospace'
                }}>
                  [SECONDARY]
                </button>
                <button style={{ 
                  backgroundColor: 'transparent', 
                  color: '#666666', 
                  padding: '8px 16px', 
                  borderRadius: '2px',
                  border: '1px solid #333333',
                  cursor: 'not-allowed',
                  fontFamily: 'monospace'
                }} disabled>
                  [DISABLED]
                </button>
              </div>
            </div>

            {/* Form Elements */}
            <div className="mb-6">
              <h3 className="font-bold mb-2" style={{ color: '#00AA00' }}>
                :: FORM_ELEMENTS ::
              </h3>
              <div className="space-y-3">
                <input 
                  type="text" 
                  placeholder="user@input:~$ " 
                  style={{ 
                    width: '100%', 
                    border: '1px solid #00AA00', 
                    padding: '8px',
                    borderRadius: '2px',
                    backgroundColor: '#000000',
                    color: '#00FF00',
                    fontFamily: 'monospace'
                  }} 
                />
                <select style={{ 
                  width: '100%', 
                  border: '1px solid #00AA00', 
                  padding: '8px',
                  borderRadius: '2px',
                  backgroundColor: '#000000',
                  color: '#00FF00',
                  fontFamily: 'monospace'
                }}>
                  <option>{'>'} SELECT_OPTION</option>
                  <option>{'>'} OPTION_1</option>
                  <option>{'>'} OPTION_2</option>
                </select>
              </div>
            </div>

            <button 
              onClick={() => applyStyle('neo')}
              style={{ 
                backgroundColor: '#00FF00', 
                color: '#000000', 
                padding: '10px 24px', 
                borderRadius: '2px',
                border: 'none',
                fontWeight: 'bold',
                fontFamily: 'monospace'
              }}
            >
              {'>'} EXECUTE_STYLE
            </button>
          </div>
        </div>
      )}

      {/* Paper Craft Style */}
      {activeProposal === 'paper' && (
        <div className="space-y-6">
          <div style={{ 
            boxShadow: '2px 2px 0px rgba(0,0,0,0.1)', 
            padding: '32px', 
            borderRadius: '12px',
            backgroundColor: '#FFFEF9',
            border: 'none'
          }}>
            <h2 className="text-lg font-bold mb-4" style={{ color: '#2C3E50' }}>
              Paper Craft Style
            </h2>
            
            {/* Sample Table */}
            <div className="mb-6">
              <h3 className="font-semibold mb-3" style={{ color: '#34495E' }}>
                Table Design
              </h3>
              <div style={{ 
                backgroundColor: 'white', 
                borderRadius: '8px', 
                boxShadow: '1px 1px 0px rgba(0,0,0,0.08)',
                overflow: 'hidden'
              }}>
                <table className="w-full">
                  <thead style={{ backgroundColor: '#F8F7F3' }}>
                    <tr>
                      <th className="text-left p-3" style={{ color: '#2C3E50', fontWeight: '500' }}>ID</th>
                      <th className="text-left p-3" style={{ color: '#2C3E50', fontWeight: '500' }}>Title</th>
                      <th className="text-left p-3" style={{ color: '#2C3E50', fontWeight: '500' }}>Status</th>
                      <th className="text-left p-3" style={{ color: '#2C3E50', fontWeight: '500' }}>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style={{ borderTop: '1px solid #F0EDE5' }}>
                      <td className="p-3" style={{ color: '#7F8C8D' }}>001</td>
                      <td className="p-3" style={{ color: '#2C3E50' }}>Sample Document</td>
                      <td className="p-3">
                        <span style={{ 
                          color: '#27AE60', 
                          backgroundColor: '#E8F8F5', 
                          padding: '4px 12px', 
                          borderRadius: '12px',
                          fontSize: '12px',
                          fontWeight: '500'
                        }}>
                          Active
                        </span>
                      </td>
                      <td className="p-3 space-x-2">
                        <button style={{ 
                          color: '#3498DB', 
                          backgroundColor: '#EBF5FB',
                          padding: '6px 12px', 
                          borderRadius: '16px',
                          border: 'none',
                          fontSize: '13px'
                        }}>
                          ✏️ Edit
                        </button>
                        <button style={{ 
                          color: '#E74C3C', 
                          backgroundColor: '#FDEDEC',
                          padding: '6px 12px', 
                          borderRadius: '16px',
                          border: 'none',
                          fontSize: '13px'
                        }}>
                          🗑 Delete
                        </button>
                        <button style={{ 
                          color: '#27AE60', 
                          backgroundColor: '#E8F8F5',
                          padding: '6px 12px', 
                          borderRadius: '16px',
                          border: 'none',
                          fontSize: '13px'
                        }}>
                          🔄 Rechunk
                        </button>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* Buttons */}
            <div className="mb-8">
              <h3 className="font-semibold mb-4" style={{ color: '#34495E' }}>
                Buttons
              </h3>
              <div className="space-x-3">
                <button style={{ 
                  backgroundColor: '#3498DB', 
                  color: 'white', 
                  padding: '10px 20px',
                  margin: '4px',
                  borderRadius: '20px',
                  border: 'none',
                  boxShadow: '1px 1px 0px rgba(0,0,0,0.1)',
                  fontWeight: '500'
                }}>
                  Primary Button
                </button>
                <button style={{ 
                  backgroundColor: 'white', 
                  color: '#3498DB', 
                  padding: '10px 20px', 
                  borderRadius: '20px',
                  border: '1px solid #BDD7ED',
                  boxShadow: '1px 1px 0px rgba(0,0,0,0.05)',
                  fontWeight: '500'
                }}>
                  Secondary Button
                </button>
                <button style={{ 
                  backgroundColor: '#ECF0F1', 
                  color: '#95A5A6', 
                  padding: '10px 20px', 
                  borderRadius: '20px',
                  border: '1px solid #D5DBDB',
                  cursor: 'not-allowed',
                  fontWeight: '500'
                }} disabled>
                  Disabled
                </button>
              </div>
            </div>

            {/* Form Elements - Refined */}
            <div className="mb-8">
              <h3 className="font-semibold mb-4" style={{ color: '#34495E' }}>
                Form Elements (Refined)
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', maxWidth: '400px' }}>
                <div>
                  <label style={{ 
                    display: 'block', 
                    marginBottom: '6px', 
                    fontSize: '13px', 
                    color: '#7F8C8D',
                    fontWeight: '500'
                  }}>
                    Document Title
                  </label>
                  <input 
                    type="text" 
                    placeholder="Enter document title..." 
                    style={{ 
                      width: '100%', 
                      border: '1px solid #E1E8ED', 
                      padding: '8px 12px',
                      borderRadius: '6px',
                      backgroundColor: 'white',
                      color: '#2C3E50',
                      fontSize: '14px',
                      transition: 'border-color 0.2s'
                    }} 
                  />
                </div>

                <div>
                  <label style={{ 
                    display: 'block', 
                    marginBottom: '6px', 
                    fontSize: '13px', 
                    color: '#7F8C8D',
                    fontWeight: '500'
                  }}>
                    Category
                  </label>
                  <select style={{ 
                    width: '100%', 
                    border: '1px solid #E1E8ED', 
                    padding: '8px 12px',
                    borderRadius: '6px',
                    backgroundColor: 'white',
                    color: '#2C3E50',
                    fontSize: '14px',
                    cursor: 'pointer',
                    appearance: 'none',
                    backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3E%3Cpath stroke='%236B7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3E%3C/svg%3E")`,
                    backgroundPosition: 'right 8px center',
                    backgroundRepeat: 'no-repeat',
                    backgroundSize: '20px',
                    paddingRight: '36px'
                  }}>
                    <option>Select category...</option>
                    <option>Documentation</option>
                    <option>Technical Spec</option>
                    <option>User Manual</option>
                  </select>
                </div>

                <div>
                  <label style={{ 
                    display: 'block', 
                    marginBottom: '6px', 
                    fontSize: '13px', 
                    color: '#7F8C8D',
                    fontWeight: '500'
                  }}>
                    Description
                  </label>
                  <textarea 
                    placeholder="Add a brief description..." 
                    rows={3}
                    style={{ 
                      width: '100%', 
                      border: '1px solid #E1E8ED', 
                      padding: '8px 12px',
                      borderRadius: '6px',
                      backgroundColor: 'white',
                      color: '#2C3E50',
                      fontSize: '14px',
                      resize: 'vertical',
                      fontFamily: 'inherit'
                    }} 
                  />
                </div>
              </div>
            </div>

            {/* Inline Form Example */}
            <div className="mb-8">
              <h3 className="font-semibold mb-4" style={{ color: '#34495E' }}>
                Inline Form Elements
              </h3>
              <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-end' }}>
                <div style={{ flex: 1 }}>
                  <label style={{ 
                    display: 'block', 
                    marginBottom: '6px', 
                    fontSize: '13px', 
                    color: '#7F8C8D',
                    fontWeight: '500'
                  }}>
                    Chunk Size
                  </label>
                  <input 
                    type="number" 
                    placeholder="1500" 
                    style={{ 
                      width: '100%', 
                      border: '1px solid #E1E8ED', 
                      padding: '8px 12px',
                      borderRadius: '6px',
                      backgroundColor: 'white',
                      color: '#2C3E50',
                      fontSize: '14px'
                    }} 
                  />
                </div>
                <div style={{ flex: 1 }}>
                  <label style={{ 
                    display: 'block', 
                    marginBottom: '6px', 
                    fontSize: '13px', 
                    color: '#7F8C8D',
                    fontWeight: '500'
                  }}>
                    Overlap
                  </label>
                  <input 
                    type="number" 
                    placeholder="200" 
                    style={{ 
                      width: '100%', 
                      border: '1px solid #E1E8ED', 
                      padding: '8px 12px',
                      borderRadius: '6px',
                      backgroundColor: 'white',
                      color: '#2C3E50',
                      fontSize: '14px'
                    }} 
                  />
                </div>
                <button style={{ 
                  backgroundColor: '#3498DB', 
                  color: 'white', 
                  padding: '8px 16px',
                  borderRadius: '6px',
                  border: 'none',
                  fontWeight: '500',
                  fontSize: '14px',
                  whiteSpace: 'nowrap'
                }}>
                  Apply
                </button>
              </div>
            </div>

            {/* Checkbox and Radio Examples */}
            <div className="mb-8">
              <h3 className="font-semibold mb-4" style={{ color: '#34495E' }}>
                Selection Controls
              </h3>
              <div style={{ display: 'flex', gap: '40px' }}>
                <div>
                  <div style={{ marginBottom: '12px' }}>
                    <label style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '8px',
                      cursor: 'pointer',
                      fontSize: '14px',
                      color: '#2C3E50'
                    }}>
                      <input type="checkbox" style={{ width: '16px', height: '16px' }} />
                      Enable auto-processing
                    </label>
                  </div>
                  <div>
                    <label style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '8px',
                      cursor: 'pointer',
                      fontSize: '14px',
                      color: '#2C3E50'
                    }}>
                      <input type="checkbox" style={{ width: '16px', height: '16px' }} />
                      Include metadata
                    </label>
                  </div>
                </div>
                <div>
                  <div style={{ marginBottom: '12px' }}>
                    <label style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '8px',
                      cursor: 'pointer',
                      fontSize: '14px',
                      color: '#2C3E50'
                    }}>
                      <input type="radio" name="strategy" style={{ width: '16px', height: '16px' }} />
                      Recursive chunking
                    </label>
                  </div>
                  <div>
                    <label style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '8px',
                      cursor: 'pointer',
                      fontSize: '14px',
                      color: '#2C3E50'
                    }}>
                      <input type="radio" name="strategy" style={{ width: '16px', height: '16px' }} />
                      Semantic chunking
                    </label>
                  </div>
                </div>
              </div>
            </div>

            <button 
              onClick={() => applyStyle('paper')}
              style={{ 
                backgroundColor: '#3498DB', 
                color: 'white', 
                padding: '12px 28px',
                margin: '8px 0',
                borderRadius: '24px',
                border: 'none',
                boxShadow: '2px 2px 0px rgba(0,0,0,0.1)',
                fontWeight: '500'
              }}
            >
              Apply Paper Craft Style
            </button>
          </div>
        </div>
      )}

      {/* Style Comparison Notes */}
      <div className="mt-8 p-4 bg-gray-50 border border-gray-300">
        <h3 className="font-bold mb-2">KEY DIFFERENCES</h3>
        <div className="space-y-2 text-sm">
          <div>
            <strong>Current (Brutalist):</strong> Sharp corners, thick 2px borders, cryptic [E][D][R] buttons, high contrast
          </div>
          <div>
            <strong>Soft Minimal:</strong> 4px radius, 1px light borders, full button text, subtle gray palette, better accessibility
          </div>
          <div>
            <strong>Neo-Terminal:</strong> 2px radius, dotted/dashed borders, icon+text buttons, green terminal colors, monospace everything
          </div>
          <div>
            <strong>Paper Craft:</strong> 8-12px radius, soft shadows instead of borders, pill buttons, warm paper tones, tactile feel
          </div>
        </div>
      </div>
    </div>
  );
}