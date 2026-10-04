"""
Interactive 3D Visualizations & WebGL Animations Module
Provides:
1. Interactive 3D Animated DNA Double Helix (Canvas 3D / WebGL) with real-time base pair rendering & mutation locus glow
2. Interactive 3D Quantum Bloch Sphere Animation with Quantum Gate Controls (H, X, Z, Ry) & Live Probability State
3. 3D Plotly Quantum Loss Landscape & Parameter Optimization Trajectory
"""
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components


def render_3d_dna_helix(height: int = 440) -> None:
    """
    Renders an interactive, animated 3D DNA Double Helix using a high-performance
    self-contained 3D Canvas rendering engine with glowing nucleotide rungs,
    interactive rotation controls, and mutation locus highlighting.
    """
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: #030712;
                color: #F8FAFC;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                overflow: hidden;
            }}
            #dna-container {{
                position: relative;
                width: 100%;
                height: {height}px;
                background: radial-gradient(circle at center, #0F172A 0%, #030712 100%);
                border-radius: 12px;
                border: 1px solid #1E293B;
                box-shadow: inset 0 0 30px rgba(0,0,0,0.8), 0 4px 20px rgba(6, 182, 212, 0.15);
            }}
            canvas {{
                display: block;
                width: 100%;
                height: 100%;
                cursor: grab;
            }}
            canvas:active {{
                cursor: grabbing;
            }}
            .dna-hud {{
                position: absolute;
                top: 14px;
                left: 16px;
                background: rgba(15, 23, 42, 0.85);
                border: 1px solid rgba(56, 189, 248, 0.3);
                padding: 8px 14px;
                border-radius: 8px;
                font-size: 12px;
                backdrop-filter: blur(8px);
                pointer-events: none;
            }}
            .dna-controls {{
                position: absolute;
                bottom: 14px;
                right: 16px;
                display: flex;
                gap: 8px;
                background: rgba(15, 23, 42, 0.85);
                border: 1px solid #334155;
                padding: 6px 10px;
                border-radius: 8px;
                backdrop-filter: blur(8px);
            }}
            .btn-ctrl {{
                background: #1E293B;
                color: #38BDF8;
                border: 1px solid #0284C7;
                padding: 4px 10px;
                border-radius: 4px;
                font-size: 11px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.2s;
            }}
            .btn-ctrl:hover {{
                background: #0284C7;
                color: #FFFFFF;
                box-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
            }}
            .legend-box {{
                position: absolute;
                bottom: 14px;
                left: 16px;
                display: flex;
                gap: 10px;
                background: rgba(15, 23, 42, 0.85);
                border: 1px solid #334155;
                padding: 6px 12px;
                border-radius: 8px;
                font-size: 11px;
                backdrop-filter: blur(8px);
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 4px;
            }}
            .dot {{
                width: 8px;
                height: 8px;
                border-radius: 50%;
                display: inline-block;
            }}
        </style>
    </head>
    <body>
        <div id="dna-container">
            <canvas id="dnaCanvas"></canvas>
            <div class="dna-hud">
                <span style="color: #38BDF8; font-weight: 700;">🧬 3D Real-Time DNA Dynamics</span><br>
                <span style="color: #94A3B8;">Double Helix Structural Model (Drag to Rotate)</span>
            </div>
            <div class="legend-box">
                <div class="legend-item"><span class="dot" style="background: #10B981;"></span> Adenine (A)</div>
                <div class="legend-item"><span class="dot" style="background: #F59E0B;"></span> Thymine (T)</div>
                <div class="legend-item"><span class="dot" style="background: #06B6D4;"></span> Guanine (G)</div>
                <div class="legend-item"><span class="dot" style="background: #6366F1;"></span> Cytosine (C)</div>
                <div class="legend-item"><span class="dot" style="background: #EF4444; box-shadow: 0 0 6px #EF4444;"></span> Mutation Locus</div>
            </div>
            <div class="dna-controls">
                <button class="btn-ctrl" id="btnToggleMut">Toggle Mutation Glow</button>
                <button class="btn-ctrl" id="btnTogglePause">Pause / Play</button>
                <button class="btn-ctrl" id="btnReset">Reset View</button>
            </div>
        </div>

        <script>
            const canvas = document.getElementById('dnaCanvas');
            const ctx = canvas.getContext('2d');
            const container = document.getElementById('dna-container');

            let width, height;
            function resize() {{
                width = canvas.width = container.clientWidth;
                height = canvas.height = container.clientHeight;
            }}
            window.addEventListener('resize', resize);
            resize();

            // DNA Simulation Parameters
            const numPairs = 36;
            const helixRadius = 75;
            const helixHeight = 360;
            const turns = 2.4;
            let rotY = 0;
            let rotX = 0.25;
            let autoRotate = true;
            let showMutation = true;

            // Mouse Interaction
            let isDragging = false;
            let prevMouseX = 0;
            let prevMouseY = 0;

            canvas.addEventListener('mousedown', (e) => {{
                isDragging = true;
                prevMouseX = e.clientX;
                prevMouseY = e.clientY;
            }});

            window.addEventListener('mouseup', () => isDragging = false);

            window.addEventListener('mousemove', (e) => {{
                if (!isDragging) return;
                const deltaX = e.clientX - prevMouseX;
                const deltaY = e.clientY - prevMouseY;
                rotY += deltaX * 0.008;
                rotX += deltaY * 0.008;
                prevMouseX = e.clientX;
                prevMouseY = e.clientY;
            }});

            document.getElementById('btnToggleMut').addEventListener('click', () => {{
                showMutation = !showMutation;
            }});

            document.getElementById('btnTogglePause').addEventListener('click', () => {{
                autoRotate = !autoRotate;
            }});

            document.getElementById('btnReset').addEventListener('click', () => {{
                rotY = 0;
                rotX = 0.25;
                autoRotate = true;
            }});

            const bases = [
                {{ name1: 'A', name2: 'T', col1: '#10B981', col2: '#F59E0B' }},
                {{ name1: 'G', name2: 'C', col1: '#06B6D4', col2: '#6366F1' }},
                {{ name1: 'T', name2: 'A', col1: '#F59E0B', col2: '#10B981' }},
                {{ name1: 'C', name2: 'G', col1: '#6366F1', col2: '#06B6D4' }},
            ];

            const mutationIndex = 18; // Middle base-pair locus

            function project(x, y, z) {{
                // 3D rotation around Y and X
                const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
                const x1 = x * cosY - z * sinY;
                const z1 = z * cosY + x * sinY;

                const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
                const y2 = y * cosX - z1 * sinX;
                const z2 = z1 * cosX + y * sinX;

                const fov = 420;
                const scale = fov / (fov + z2 + 200);
                return {{
                    x: width / 2 + x1 * scale,
                    y: height / 2 + y2 * scale,
                    scale: scale,
                    depth: z2
                }};
            }}

            function draw() {{
                ctx.clearRect(0, 0, width, height);

                if (autoRotate && !isDragging) {{
                    rotY += 0.012;
                }}

                const renderQueue = [];

                for (let i = 0; i < numPairs; i++) {{
                    const t = (i / (numPairs - 1)) - 0.5;
                    const angle = t * turns * Math.PI * 2;
                    const y = t * helixHeight;

                    const x1 = Math.cos(angle) * helixRadius;
                    const z1 = Math.sin(angle) * helixRadius;

                    const x2 = Math.cos(angle + Math.PI) * helixRadius;
                    const z2 = Math.sin(angle + Math.PI) * helixRadius;

                    const p1 = project(x1, y, z1);
                    const p2 = project(x2, y, z2);

                    const pairInfo = bases[i % bases.length];
                    const isMut = (i === mutationIndex && showMutation);

                    renderQueue.push({{
                        type: 'rung',
                        depth: (p1.depth + p2.depth) / 2,
                        p1: p1,
                        p2: p2,
                        isMut: isMut,
                        pair: pairInfo,
                        idx: i
                    }});

                    renderQueue.push({{
                        type: 'node',
                        depth: p1.depth,
                        pos: p1,
                        color: isMut ? '#EF4444' : pairInfo.col1,
                        isMut: isMut
                    }});

                    renderQueue.push({{
                        type: 'node',
                        depth: p2.depth,
                        pos: p2,
                        color: isMut ? '#F87171' : pairInfo.col2,
                        isMut: isMut
                    }});
                }}

                // Sort by depth for correct 3D occlusion
                renderQueue.sort((a, b) => b.depth - a.depth);

                // Draw connecting backbone strands
                ctx.save();
                for (let i = 0; i < renderQueue.length; i++) {{
                    const item = renderQueue[i];

                    if (item.type === 'rung') {{
                        ctx.beginPath();
                        ctx.moveTo(item.p1.x, item.p1.y);
                        ctx.lineTo(item.p2.x, item.p2.y);
                        ctx.lineWidth = Math.max(1, 3.5 * item.p1.scale);

                        if (item.isMut) {{
                            ctx.strokeStyle = '#EF4444';
                            ctx.shadowColor = '#EF4444';
                            ctx.shadowBlur = 15;
                        }} else {{
                            ctx.strokeStyle = 'rgba(100, 116, 139, 0.45)';
                            ctx.shadowColor = 'transparent';
                            ctx.shadowBlur = 0;
                        }}
                        ctx.stroke();

                        // Midpoint hydrogen bond marker
                        const midX = (item.p1.x + item.p2.x) / 2;
                        const midY = (item.p1.y + item.p2.y) / 2;
                        ctx.beginPath();
                        ctx.arc(midX, midY, 2.5 * item.p1.scale, 0, Math.PI * 2);
                        ctx.fillStyle = item.isMut ? '#FCA5A5' : 'rgba(255, 255, 255, 0.6)';
                        ctx.fill();

                        if (item.isMut) {{
                            ctx.font = 'bold 11px monospace';
                            ctx.fillStyle = '#EF4444';
                            ctx.fillText('◄ MUTATION LOCUS', item.p2.x + 12, item.p2.y + 4);
                        }}
                    }} else if (item.type === 'node') {{
                        ctx.beginPath();
                        const r = Math.max(2, (item.isMut ? 8 : 5.5) * item.pos.scale);
                        ctx.arc(item.pos.x, item.pos.y, r, 0, Math.PI * 2);
                        ctx.fillStyle = item.color;
                        if (item.isMut) {{
                            ctx.shadowColor = '#EF4444';
                            ctx.shadowBlur = 18;
                        }} else {{
                            ctx.shadowColor = item.color;
                            ctx.shadowBlur = 6;
                        }}
                        ctx.fill();
                    }}
                }}
                ctx.restore();

                requestAnimationFrame(draw);
            }}

            draw();
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height)


def render_3d_quantum_bloch_sphere(height: int = 440) -> None:
    """
    Renders an interactive 3D Quantum Bloch Sphere animation with real-time statevector
    rotations, gate operations (H, X, Z, Ry), and live computational basis probabilities.
    """
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: #030712;
                color: #F8FAFC;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                overflow: hidden;
            }}
            #bloch-container {{
                position: relative;
                width: 100%;
                height: {height}px;
                background: radial-gradient(circle at center, #0F172A 0%, #030712 100%);
                border-radius: 12px;
                border: 1px solid #1E293B;
                box-shadow: inset 0 0 30px rgba(0,0,0,0.8), 0 4px 20px rgba(99, 102, 241, 0.15);
            }}
            canvas {{
                display: block;
                width: 100%;
                height: 100%;
                cursor: grab;
            }}
            canvas:active {{
                cursor: grabbing;
            }}
            .bloch-hud {{
                position: absolute;
                top: 14px;
                left: 16px;
                background: rgba(15, 23, 42, 0.85);
                border: 1px solid rgba(99, 102, 241, 0.35);
                padding: 8px 14px;
                border-radius: 8px;
                font-size: 12px;
                backdrop-filter: blur(8px);
                pointer-events: none;
            }}
            .state-readout {{
                position: absolute;
                bottom: 14px;
                left: 16px;
                background: rgba(15, 23, 42, 0.85);
                border: 1px solid #334155;
                padding: 8px 14px;
                border-radius: 8px;
                font-size: 12px;
                font-family: monospace;
                backdrop-filter: blur(8px);
            }}
            .gate-controls {{
                position: absolute;
                top: 14px;
                right: 16px;
                display: flex;
                flex-direction: column;
                gap: 6px;
                background: rgba(15, 23, 42, 0.85);
                border: 1px solid #334155;
                padding: 8px 10px;
                border-radius: 8px;
                backdrop-filter: blur(8px);
            }}
            .btn-gate {{
                background: #1E293B;
                color: #A5B4FC;
                border: 1px solid #6366F1;
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 11px;
                cursor: pointer;
                font-weight: 700;
                transition: all 0.2s;
                text-align: left;
            }}
            .btn-gate:hover {{
                background: #6366F1;
                color: #FFFFFF;
                box-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
            }}
        </style>
    </head>
    <body>
        <div id="bloch-container">
            <canvas id="blochCanvas"></canvas>
            <div class="bloch-hud">
                <span style="color: #A5B4FC; font-weight: 700;">⚛️ 3D Quantum State Bloch Sphere</span><br>
                <span style="color: #94A3B8;">|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩</span>
            </div>
            <div class="state-readout" id="stateReadout">
                <span style="color: #38BDF8;">P(|0⟩): 1.000</span> | <span style="color: #EC4899;">P(|1⟩): 0.000</span><br>
                <span style="color: #94A3B8;">θ = 0.000 rad | φ = 0.000 rad</span>
            </div>
            <div class="gate-controls">
                <div style="font-size: 10px; color: #94A3B8; font-weight: 700; margin-bottom: 2px;">APPLY QUANTUM GATES:</div>
                <button class="btn-gate" id="btnHadamard">H: Hadamard (|0⟩ ➜ |+⟩)</button>
                <button class="btn-gate" id="btnPauliX">X: Bit Flip (|0⟩ ➜ |1⟩)</button>
                <button class="btn-gate" id="btnPauliZ">Z: Phase Flip</button>
                <button class="btn-gate" id="btnRy">Ry(π/4): DNA PQC Rotation</button>
                <button class="btn-gate" id="btnReset">Reset to Ground State |0⟩</button>
            </div>
        </div>

        <script>
            const canvas = document.getElementById('blochCanvas');
            const ctx = canvas.getContext('2d');
            const container = document.getElementById('bloch-container');

            let width, height;
            function resize() {{
                width = canvas.width = container.clientWidth;
                height = canvas.height = container.clientHeight;
            }}
            window.addEventListener('resize', resize);
            resize();

            // Quantum State Parameters: theta [0, pi], phi [0, 2pi]
            let theta = 0.0;
            let phi = 0.0;
            let targetTheta = 0.0;
            let targetPhi = 0.0;

            // View Orientation
            let viewRotY = -0.5;
            let viewRotX = 0.35;
            let isDragging = false;
            let prevMouseX = 0;
            let prevMouseY = 0;

            canvas.addEventListener('mousedown', (e) => {{
                isDragging = true;
                prevMouseX = e.clientX;
                prevMouseY = e.clientY;
            }});

            window.addEventListener('mouseup', () => isDragging = false);

            window.addEventListener('mousemove', (e) => {{
                if (!isDragging) return;
                const deltaX = e.clientX - prevMouseX;
                const deltaY = e.clientY - prevMouseY;
                viewRotY += deltaX * 0.008;
                viewRotX += deltaY * 0.008;
                prevMouseX = e.clientX;
                prevMouseY = e.clientY;
            }});

            document.getElementById('btnHadamard').addEventListener('click', () => {{
                targetTheta = Math.PI / 2;
                targetPhi = 0.0;
            }});

            document.getElementById('btnPauliX').addEventListener('click', () => {{
                targetTheta = (targetTheta < Math.PI / 2) ? Math.PI : 0.0;
            }});

            document.getElementById('btnPauliZ').addEventListener('click', () => {{
                targetPhi += Math.PI;
            }});

            document.getElementById('btnRy').addEventListener('click', () => {{
                targetTheta = (targetTheta + Math.PI / 4) % (Math.PI + 0.01);
                targetPhi = (targetPhi + Math.PI / 6) % (Math.PI * 2);
            }});

            document.getElementById('btnReset').addEventListener('click', () => {{
                targetTheta = 0.0;
                targetPhi = 0.0;
            }});

            const radius = 110;

            function project(x, y, z) {{
                const cosY = Math.cos(viewRotY), sinY = Math.sin(viewRotY);
                const x1 = x * cosY - z * sinY;
                const z1 = z * cosY + x * sinY;

                const cosX = Math.cos(viewRotX), sinX = Math.sin(viewRotX);
                const y2 = y * cosX - z1 * sinX;
                const z2 = z1 * cosX + y * sinX;

                const fov = 400;
                const scale = fov / (fov + z2 + 200);
                return {{
                    x: width / 2 + x1 * scale,
                    y: height / 2 + y2 * scale,
                    scale: scale,
                    depth: z2
                }};
            }}

            function draw() {{
                ctx.clearRect(0, 0, width, height);

                // Smooth interpolation to target quantum state
                theta += (targetTheta - theta) * 0.08;
                phi += (targetPhi - phi) * 0.08;

                // Update probabilities
                const p0 = Math.cos(theta / 2) ** 2;
                const p1 = Math.sin(theta / 2) ** 2;
                document.getElementById('stateReadout').innerHTML =
                    `<span style="color: #38BDF8;">P(|0⟩): ${{p0.toFixed(3)}}</span> | <span style="color: #EC4899;">P(|1⟩): ${{p1.toFixed(3)}}</span><br>` +
                    `<span style="color: #94A3B8;">θ = ${{theta.toFixed(3)}} rad | φ = ${{phi.toFixed(3)}} rad</span>`;

                // Draw Coordinate Axes
                const axes = [
                    {{ name: 'X', start: [-radius * 1.25, 0, 0], end: [radius * 1.25, 0, 0], col: '#475569' }},
                    {{ name: 'Y', start: [0, -radius * 1.25, 0], end: [0, radius * 1.25, 0], col: '#475569' }},
                    {{ name: 'Z', start: [0, 0, -radius * 1.25], end: [0, 0, radius * 1.25], col: '#64748B' }},
                ];

                axes.forEach(ax => {{
                    const pStart = project(ax.start[0], ax.start[1], ax.start[2]);
                    const pEnd = project(ax.end[0], ax.end[1], ax.end[2]);
                    ctx.beginPath();
                    ctx.moveTo(pStart.x, pStart.y);
                    ctx.lineTo(pEnd.x, pEnd.y);
                    ctx.strokeStyle = ax.col;
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }});

                // Draw Equator Circle
                ctx.beginPath();
                for (let a = 0; a <= Math.PI * 2; a += 0.1) {{
                    const x = Math.cos(a) * radius;
                    const y = Math.sin(a) * radius;
                    const p = project(x, y, 0);
                    if (a === 0) ctx.moveTo(p.x, p.y);
                    else ctx.lineTo(p.x, p.y);
                }}
                ctx.strokeStyle = 'rgba(6, 182, 212, 0.4)';
                ctx.lineWidth = 1.5;
                ctx.stroke();

                // Draw Meridian Circles
                ctx.beginPath();
                for (let a = 0; a <= Math.PI * 2; a += 0.1) {{
                    const x = Math.cos(a) * radius;
                    const z = Math.sin(a) * radius;
                    const p = project(x, 0, z);
                    if (a === 0) ctx.moveTo(p.x, p.y);
                    else ctx.lineTo(p.x, p.y);
                }}
                ctx.strokeStyle = 'rgba(99, 102, 241, 0.25)';
                ctx.stroke();

                // Compute Statevector in Cartesian Coordinates
                // |0> is at Z = +radius (North), |1> is at Z = -radius (South)
                const stateX = radius * Math.sin(theta) * Math.cos(phi);
                const stateY = radius * Math.sin(theta) * Math.sin(phi);
                const stateZ = radius * Math.cos(theta);

                const pOrigin = project(0, 0, 0);
                const pState = project(stateX, stateY, stateZ);

                // Draw Statevector Arrow
                ctx.save();
                ctx.beginPath();
                ctx.moveTo(pOrigin.x, pOrigin.y);
                ctx.lineTo(pState.x, pState.y);
                ctx.strokeStyle = '#06B6D4';
                ctx.shadowColor = '#06B6D4';
                ctx.shadowBlur = 12;
                ctx.lineWidth = 3.5;
                ctx.stroke();

                // Draw Statevector Endpoint Sphere
                ctx.beginPath();
                ctx.arc(pState.x, pState.y, 7 * pState.scale, 0, Math.PI * 2);
                ctx.fillStyle = '#FFFFFF';
                ctx.shadowColor = '#38BDF8';
                ctx.shadowBlur = 16;
                ctx.fill();
                ctx.restore();

                // Draw |0> and |1> Labels
                const pNorth = project(0, 0, radius * 1.15);
                const pSouth = project(0, 0, -radius * 1.15);
                ctx.font = 'bold 12px monospace';
                ctx.fillStyle = '#34D399';
                ctx.fillText('|0⟩ (WT)', pNorth.x + 8, pNorth.y);
                ctx.fillStyle = '#F87171';
                ctx.fillText('|1⟩ (MUT)', pSouth.x + 8, pSouth.y);

                requestAnimationFrame(draw);
            }}

            draw();
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height)


def create_quantum_loss_landscape_3d() -> go.Figure:
    """
    Generates an interactive 3D surface plot depicting the Quantum Variational
    Cost/Loss Landscape E(theta_1, theta_2) with simulated optimization trajectories.
    """
    th1 = np.linspace(-np.pi, np.pi, 45)
    th2 = np.linspace(-np.pi, np.pi, 45)
    T1, T2 = np.meshgrid(th1, th2)

    # Quantum expectation landscape with interference ripples
    Z = (
        0.5 * (1 - np.cos(T1) * np.cos(T2))
        + 0.18 * np.sin(2 * T1) * np.sin(2 * T2)
        + 0.08 * np.cos(3 * T1)
    )

    # Simulated COBYLA / Adam Optimization Path
    traj_th1 = [-2.4, -1.8, -1.2, -0.6, -0.2, 0.0]
    traj_th2 = [2.2, 1.6, 1.1, 0.5, 0.15, 0.0]
    traj_z = [
        0.5 * (1 - np.cos(t1) * np.cos(t2))
        + 0.18 * np.sin(2 * t1) * np.sin(2 * t2)
        + 0.08 * np.cos(3 * t1) + 0.03
        for t1, t2 in zip(traj_th1, traj_th2)
    ]

    fig = go.Figure()

    # 3D Surface
    fig.add_trace(go.Surface(
        z=Z, x=T1, y=T2,
        colorscale='Viridis',
        opacity=0.88,
        showscale=False,
        name="Quantum Energy Surface ⟨H(θ)⟩",
    ))

    # Optimization Trajectory
    fig.add_trace(go.Scatter3d(
        x=traj_th1, y=traj_th2, z=traj_z,
        mode='lines+markers',
        line=dict(color="#EF4444", width=5),
        marker=dict(size=6, color="#FCD34D", symbol="circle"),
        name="Optimizer Gradient Path",
    ))

    # Global Optimum Star
    fig.add_trace(go.Scatter3d(
        x=[0.0], y=[0.0], z=[traj_z[-1]],
        mode='markers+text',
        marker=dict(size=9, color="#10B981", symbol="diamond"),
        text=["Optimal Ground State"],
        textposition="top center",
        textfont=dict(color="#6EE7B7", size=11),
        name="Ground State Minimum",
    ))

    fig.update_layout(
        title=dict(text="⚛️ 3D Parameterized Quantum Circuit (PQC) Energy Landscape & Optimization Trajectory", font=dict(color="#38BDF8", size=14)),
        scene=dict(
            xaxis=dict(title="Ansatz Rotation θ₁", gridcolor="#334155", tickfont=dict(color="#CBD5E1")),
            yaxis=dict(title="Ansatz Rotation θ₂", gridcolor="#334155", tickfont=dict(color="#CBD5E1")),
            zaxis=dict(title="Loss / Energy ⟨H(θ)⟩", gridcolor="#334155", tickfont=dict(color="#CBD5E1")),
            bgcolor="rgba(15, 23, 42, 0.7)",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.1)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(color="#CBD5E1", size=10),
        ),
        height=450,
    )
    return fig
