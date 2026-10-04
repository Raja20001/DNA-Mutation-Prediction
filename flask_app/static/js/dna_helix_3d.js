/**
 * Interactive 3D DNA Double Helix Animation Engine
 * Pure Canvas 3D hardware-accelerated rendering with interactive mouse controls
 */
(function() {
    function initDNAHelix() {
        const canvas = document.getElementById("dnaCanvas");
        if (!canvas) return;

        const ctx = canvas.getContext("2d");
        let width = canvas.clientWidth;
        let height = canvas.clientHeight;
        canvas.width = width;
        canvas.height = height;

        window.addEventListener("resize", () => {
            width = canvas.clientWidth;
            height = canvas.clientHeight;
            canvas.width = width;
            canvas.height = height;
        });

        // State parameters
        let angle = 0;
        let rotSpeed = 0.018;
        let autoRotate = true;
        let showMutation = true;
        let isDragging = false;
        let lastMouseX = 0;
        let lastMouseY = 0;
        let pitch = 0.15; // Vertical tilt angle

        const numNodes = 42;
        const radius = 95;
        const verticalSpread = height * 0.95;
        const yStart = -verticalSpread / 2;
        const yStep = verticalSpread / numNodes;

        // Base colors (A, T, G, C)
        const baseColors = [
            { name: "Adenine", color: "#10B981" },   // Emerald
            { name: "Thymine", color: "#EF4444" },   // Ruby Red
            { name: "Guanine", color: "#06B6D4" },   // Cyan
            { name: "Cytosine", color: "#F59E0B" }   // Amber
        ];

        // Drag controls
        canvas.addEventListener("mousedown", (e) => {
            isDragging = true;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        });
        window.addEventListener("mouseup", () => {
            isDragging = false;
        });
        window.addEventListener("mousemove", (e) => {
            if (!isDragging) return;
            const dx = e.clientX - lastMouseX;
            const dy = e.clientY - lastMouseY;
            angle += dx * 0.01;
            pitch += dy * 0.005;
            pitch = Math.max(-0.6, Math.min(0.6, pitch));
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        });

        // Button / UI binding
        const btnRotate = document.getElementById("toggleRotate");
        if (btnRotate) {
            btnRotate.addEventListener("click", () => {
                autoRotate = !autoRotate;
                btnRotate.textContent = autoRotate ? "⏸️ Pause Auto-Rotate" : "▶️ Resume Auto-Rotate";
            });
        }
        const btnMut = document.getElementById("toggleMutPulse");
        if (btnMut) {
            btnMut.addEventListener("click", () => {
                showMutation = !showMutation;
                btnMut.textContent = showMutation ? "🔴 Mutation Pulse: ON" : "⚪ Mutation Pulse: OFF";
            });
        }
        const speedSlider = document.getElementById("dnaSpeed");
        if (speedSlider) {
            speedSlider.addEventListener("input", (e) => {
                rotSpeed = parseFloat(e.target.value);
            });
        }

        // Render Loop
        function render() {
            ctx.clearRect(0, 0, width, height);

            const cx = width / 2;
            const cy = height / 2;

            if (autoRotate && !isDragging) {
                angle += rotSpeed;
            }

            const itemsToDraw = [];

            for (let i = 0; i < numNodes; i++) {
                const nodeAngle = angle + (i * 0.28);
                const yPos = yStart + (i * yStep);

                // Anti-parallel Strand 1
                const x1 = Math.cos(nodeAngle) * radius;
                const z1 = Math.sin(nodeAngle) * radius;
                // Strand 2 (180 deg shifted)
                const x2 = Math.cos(nodeAngle + Math.PI) * radius;
                const z2 = Math.sin(nodeAngle + Math.PI) * radius;

                // 3D Perspective Projection with pitch
                const fov = 400;
                const cosP = Math.cos(pitch);
                const sinP = Math.sin(pitch);

                const yProj1 = yPos * cosP - z1 * sinP;
                const zProj1 = yPos * sinP + z1 * cosP + 450;
                const scale1 = fov / zProj1;
                const screenX1 = cx + x1 * scale1;
                const screenY1 = cy + yProj1 * scale1;

                const yProj2 = yPos * cosP - z2 * sinP;
                const zProj2 = yPos * sinP + z2 * cosP + 450;
                const scale2 = fov / zProj2;
                const screenX2 = cx + x2 * scale2;
                const screenY2 = cy + yProj2 * scale2;

                const basePair = baseColors[i % baseColors.length];
                const isMutantLocus = (i === Math.floor(numNodes / 2));

                itemsToDraw.push({
                    type: "rung",
                    x1: screenX1, y1: screenY1, z1: zProj1,
                    x2: screenX2, y2: screenY2, z2: zProj2,
                    avgZ: (zProj1 + zProj2) / 2,
                    color: basePair.color,
                    isMutant: isMutantLocus && showMutation,
                });

                itemsToDraw.push({
                    type: "node",
                    x: screenX1, y: screenY1, z: zProj1,
                    scale: scale1,
                    color: "#38BDF8", // Cyan backbone
                });

                itemsToDraw.push({
                    type: "node",
                    x: screenX2, y: screenY2, z: zProj2,
                    scale: scale2,
                    color: "#818CF8", // Purple backbone
                });
            }

            // Depth sorting (painter's algorithm)
            itemsToDraw.sort((a, b) => (b.avgZ || b.z) - (a.avgZ || a.z));

            // Draw objects
            itemsToDraw.forEach(item => {
                if (item.type === "rung") {
                    ctx.beginPath();
                    ctx.moveTo(item.x1, item.y1);
                    ctx.lineTo(item.x2, item.y2);
                    ctx.lineWidth = item.isMutant ? 4 : 2;
                    ctx.strokeStyle = item.isMutant ? "#EF4444" : item.color;
                    if (item.isMutant) {
                        ctx.shadowColor = "#EF4444";
                        ctx.shadowBlur = 18;
                    } else {
                        ctx.shadowBlur = 0;
                    }
                    ctx.stroke();
                    ctx.shadowBlur = 0;

                    // Midpoint base pair marker
                    const midX = (item.x1 + item.x2) / 2;
                    const midY = (item.y1 + item.y2) / 2;
                    ctx.beginPath();
                    ctx.arc(midX, midY, item.isMutant ? 6 : 3, 0, Math.PI * 2);
                    ctx.fillStyle = item.isMutant ? "#EF4444" : item.color;
                    ctx.fill();
                } else if (item.type === "node") {
                    const radiusNode = Math.max(2, 5 * item.scale);
                    ctx.beginPath();
                    ctx.arc(item.x, item.y, radiusNode, 0, Math.PI * 2);
                    ctx.fillStyle = item.color;
                    ctx.shadowColor = item.color;
                    ctx.shadowBlur = 8;
                    ctx.fill();
                    ctx.shadowBlur = 0;
                }
            });

            requestAnimationFrame(render);
        }

        render();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initDNAHelix);
    } else {
        initDNAHelix();
    }
})();
