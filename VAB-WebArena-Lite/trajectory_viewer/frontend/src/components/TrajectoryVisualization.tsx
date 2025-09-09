import React, { useEffect, useRef, useState } from 'react';
import { Network } from 'vis-network';
import { DataSet } from 'vis-data';
import { TrajectoryDetail, NodeInfo } from '../types';
import { trajectoryApi } from '../api';
import { X, Maximize2, Minimize2, RotateCcw } from 'lucide-react';

interface TrajectoryVisualizationProps {
  trajectory: TrajectoryDetail | null;
  onClose: () => void;
}

const TrajectoryVisualization: React.FC<TrajectoryVisualizationProps> = ({
  trajectory,
  onClose
}) => {
  const networkRef = useRef<HTMLDivElement>(null);
  const networkInstanceRef = useRef<Network | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [selectedNode, setSelectedNode] = useState<NodeInfo | null>(null);
  const [showScreenshot, setShowScreenshot] = useState(false);

  useEffect(() => {
    if (!trajectory || !networkRef.current) return;

    // Create nodes and edges
    const nodes = new DataSet();
    const edges = new DataSet();

    // Add nodes
    trajectory.nodes.forEach((node) => {
      let color = '#e1e5e9';
      let borderColor = '#9ca3af';
      let shape = 'box';
      let label = node.node_id;
      let fontColor = '#1f2937';

      // Determine node type and selection status based on status field
      let nodeType = 'unknown';
      let isSelected = false;
      
      if (node.node_id === 'root' || node.node_id.startsWith('root_')) {
        nodeType = 'root';
      } else if (node.node_id.startsWith('state_') && !node.node_id.includes('candidate')) {
        nodeType = 'state';
      } else if (node.status === 'candidate') {
        nodeType = 'candidate';
        isSelected = false;
      } else if (node.status === 'selected') {
        nodeType = 'candidate';
        isSelected = true;
      }


      if (nodeType === 'root') {
        color = '#1e40af'; // 深蓝色
        borderColor = '#1e3a8a';
        label = `ROOT\n${trajectory.intent}`;
        shape = 'ellipse';
        fontColor = '#ffffff';
      } else if (nodeType === 'state') {
        color = '#059669'; // 绿色
        borderColor = '#047857';
        label = `State\n${node.node_id.split('_')[1] || '0'}`;
        fontColor = '#ffffff';
      } else if (nodeType === 'candidate') {
        if (isSelected) {
          color = '#059669'; // 绿色 - 被选中的候选
          borderColor = '#047857';
          fontColor = '#ffffff';
        } else {
          color = '#9ca3af'; // 灰色 - 未选中的候选
          borderColor = '#6b7280';
          fontColor = '#ffffff';
        }
        label = `Candidate\n${node.node_id.split('_').pop() || ''}`;
      }

      // Determine level for hierarchical layout
      let level = 0;
      if (nodeType === 'root') {
        level = 0;
      } else if (nodeType === 'state') {
        // State nodes: level = step * 2 + 1
        level = (node.step || 0) * 2 + 1;
      } else if (nodeType === 'candidate') {
        // Candidate nodes: level = step * 2 + 2 (right after their state)
        const parentState = trajectory.nodes.find(n => 
          n.node_id.startsWith('state_') && 
          !n.node_id.includes('candidate') && 
          n.candidates && 
          n.candidates.includes(node.node_id)
        );
        level = (parentState?.step || 0) * 2 + 2;
      }

      (nodes as any).add({
        id: node.node_id,
        label,
        level: level,
        color: {
          background: color,
          border: borderColor,
          highlight: {
            background: color,
            border: borderColor
          },
          hover: {
            background: color,
            border: borderColor
          }
        },
        shape,
        font: {
          size: 12,
          color: fontColor,
          face: 'Arial, sans-serif',
          bold: node.node_type === 'root' || node.status === 'selected'
        },
        borderWidth: 3,
        shadow: {
          enabled: true,
          color: 'rgba(0,0,0,0.2)',
          size: 5,
          x: 2,
          y: 2
        }
      });
    });

    // Add edges following trajectory_tree.py logic
    // 1. Root → First State
    const firstState = trajectory.nodes.find(n => 
      n.node_id.startsWith('state_') && !n.node_id.includes('candidate')
    );
    if (firstState) {
      (edges as any).add({
        from: 'root',
        to: firstState.node_id,
        color: {
          color: '#059669',
          highlight: '#059669',
          hover: '#059669'
        },
        width: 3,
        dashes: false,
        shadow: {
          enabled: true,
          color: 'rgba(0,0,0,0.1)',
          size: 3
        }
      });
    }

    // 2. State → Candidates (all candidates)
    trajectory.nodes.forEach((node) => {
      if (node.node_id.startsWith('state_') && !node.node_id.includes('candidate') && node.candidates) {
        node.candidates.forEach((candidateId) => {
          const candidate = trajectory.nodes.find(n => n.node_id === candidateId);
          const isSelected = candidate && candidate.status === 'selected';
          
          (edges as any).add({
            from: node.node_id,
            to: candidateId,
            color: {
              color: isSelected ? '#059669' : '#9ca3af',
              highlight: isSelected ? '#059669' : '#6b7280',
              hover: isSelected ? '#059669' : '#6b7280'
            },
            width: isSelected ? 3 : 1,
            dashes: isSelected ? false : true,
            shadow: {
              enabled: true,
              color: 'rgba(0,0,0,0.1)',
              size: 2
            }
          });
        });
      }
    });

    // 3. Selected Candidate → Next State
    trajectory.nodes.forEach((node) => {
      if (node.status === 'selected') {
        // Find the next state (step + 1)
        const currentState = trajectory.nodes.find(n => 
          n.node_id.startsWith('state_') && 
          !n.node_id.includes('candidate') && 
          n.candidates && 
          n.candidates.includes(node.node_id)
        );
        
        if (currentState) {
          const nextState = trajectory.nodes.find(n => 
            n.node_id.startsWith('state_') && 
            !n.node_id.includes('candidate') && 
            n.step === (currentState.step || 0) + 1
          );
          
          if (nextState) {
            (edges as any).add({
              from: node.node_id,
              to: nextState.node_id,
              color: {
                color: '#059669',
                highlight: '#059669',
                hover: '#059669'
              },
              width: 4,
              dashes: false,
              shadow: {
                enabled: true,
                color: 'rgba(0,0,0,0.1)',
                size: 3
              }
            });
          }
        }
      }
    });

    // Network options
    const options = {
      layout: {
        hierarchical: {
          enabled: true,
          direction: 'UD' as const,
          sortMethod: 'directed' as const,
          levelSeparation: 250,
          nodeSpacing: 100,
          treeSpacing: 300,
          blockShifting: true,
          edgeMinimization: true,
          parentCentralization: true
        }
      },
      physics: {
        enabled: false
      },
      interaction: {
        hover: true,
        selectConnectedEdges: false,
        tooltipDelay: 200,
        hideEdgesOnDrag: false
      },
      nodes: {
        borderWidth: 3,
        shadow: {
          enabled: true,
          color: 'rgba(0,0,0,0.2)',
          size: 5,
          x: 2,
          y: 2
        },
        margin: {
          top: 10,
          right: 10,
          bottom: 10,
          left: 10
        },
        scaling: {
          min: 10,
          max: 30
        }
      },
      edges: {
        shadow: {
          enabled: true,
          color: 'rgba(0,0,0,0.1)',
          size: 3
        },
        smooth: {
          enabled: true,
          type: 'continuous' as const,
          roundness: 0.5
        },
        arrows: {
          to: {
            enabled: true,
            scaleFactor: 0.8
          }
        }
      }
    } as any;

    // Create network
    const network = new Network(networkRef.current, { nodes: nodes as any, edges: edges as any }, options);
    networkInstanceRef.current = network;

    // Add click event listener
    network.on('click', (params) => {
      if (params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        const node = trajectory.nodes.find(n => n.node_id === nodeId);
        if (node) {
          setSelectedNode(node);
          if (node.screenshot_path) {
            setShowScreenshot(true);
          }
        }
      }
    });

    return () => {
      if (networkInstanceRef.current) {
        networkInstanceRef.current.destroy();
        networkInstanceRef.current = null;
      }
    };
  }, [trajectory]);

  const handleFullscreen = () => {
    if (!document.fullscreenElement) {
      networkRef.current?.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const handleReset = () => {
    if (networkInstanceRef.current) {
      networkInstanceRef.current.fit();
    }
  };

  if (!trajectory) return null;

  return (
    <div className={`trajectory-visualization ${isFullscreen ? 'fullscreen' : ''}`}>
      <div className="visualization-header">
        <div className="header-left">
          <h3>{trajectory.intent}</h3>
          <span className="run-id">{trajectory.run_id}</span>
        </div>
        <div className="header-right">
          <button onClick={handleReset} className="icon-button" title="Reset view">
            <RotateCcw size={16} />
          </button>
          <button onClick={handleFullscreen} className="icon-button" title="Toggle fullscreen">
            {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
          <button onClick={onClose} className="icon-button" title="Close">
            <X size={16} />
          </button>
        </div>
      </div>
      
      <div className="visualization-content">
        <div className="network-container" ref={networkRef} />
        
        {/* Legend */}
        <div className="legend">
          <h4>Legend</h4>
          <div className="legend-items">
            <div className="legend-item">
              <div className="legend-color" style={{backgroundColor: '#1e40af'}}></div>
              <span>Root Node</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{backgroundColor: '#059669'}}></div>
              <span>State & Selected</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{backgroundColor: '#9ca3af'}}></div>
              <span>Other Candidates</span>
            </div>
          </div>
        </div>
        
        {selectedNode && (
          <div className="node-details">
            <h4>Node Details</h4>
            <div className="detail-item">
              <strong>ID:</strong> {selectedNode.node_id}
            </div>
            <div className="detail-item">
              <strong>Type:</strong> {selectedNode.node_type}
            </div>
            {selectedNode.step !== null && (
              <div className="detail-item">
                <strong>Step:</strong> {selectedNode.step}
              </div>
            )}
            {selectedNode.url && (
              <div className="detail-item">
                <strong>URL:</strong> {selectedNode.url}
              </div>
            )}
            {selectedNode.thought && (
              <div className="detail-item">
                <strong>Thought:</strong> {selectedNode.thought}
              </div>
            )}
            {selectedNode.meaning && (
              <div className="detail-item">
                <strong>Meaning:</strong> {selectedNode.meaning}
              </div>
            )}
            {selectedNode.action && (
              <div className="detail-item">
                <strong>Action:</strong> {selectedNode.action}
              </div>
            )}
            {selectedNode.status && (
              <div className="detail-item">
                <strong>Status:</strong> {selectedNode.status}
              </div>
            )}
            {selectedNode.screenshot_path && (
              <div className="detail-item">
                <button 
                  onClick={() => setShowScreenshot(!showScreenshot)}
                  className="screenshot-button"
                >
                  {showScreenshot ? 'Hide' : 'Show'} Screenshot
                </button>
              </div>
            )}
          </div>
        )}
      </div>
      
      {showScreenshot && selectedNode?.screenshot_path && (
        <div className="screenshot-modal">
          <div className="screenshot-content">
            <div className="screenshot-header">
              <h4>Screenshot - {selectedNode.node_id}</h4>
              <button onClick={() => setShowScreenshot(false)} className="close-button">
                <X size={20} />
              </button>
            </div>
            <div className="screenshot-image">
              <img 
                src={trajectoryApi.getScreenshotUrl(selectedNode.screenshot_path.split('/').pop() || '')}
                alt={`Screenshot for ${selectedNode.node_id}`}
                onError={(e) => {
                  const target = e.target as HTMLImageElement;
                  target.style.display = 'none';
                  target.nextElementSibling?.classList.remove('hidden');
                }}
              />
              <div className="screenshot-error hidden">
                <p>Screenshot not found</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TrajectoryVisualization;
