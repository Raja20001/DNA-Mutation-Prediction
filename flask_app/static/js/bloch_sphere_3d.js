/**
 * Interactive 3D Quantum Bloch Sphere Simulator Engine 2.0
 * Advanced Quantum Information Geometry, Dynamic Gate Operations & Real-Time Trajectory Tracking
 */
(function() {
    function initBlochSphere() {
        const canvas = document.getElementById("blochCanvas");
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

        // Bloch state angles: |psi> = cos(theta/2)|0> + e^(i*phi)*sin(theta/2)|1>
        let theta = 0.35; // Near |0>
        let phi = 0.0;
        let targetTheta = theta;
        let targetPhi = phi;

        let rotX = 0.35;
        let rotY = 0.45;
        let isDragging = false;
        let lastMouseX = 0;
        let lastMouseY = 0;
        let precessionSpeed = 0.012;
        let trail = []; // Historical trajectory points

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
            rotY += dx * 0.008;
            rotX += dy * 0.008;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        });

        // Advanced Gate Transformations
        function applyGate(gate) {
            if (gate === "H") {
                // Hadamard maps |0> to (|0>+|1>)/sqrt(2) => theta = pi/2, phi = 0
                targetTheta = Math.PI / 2;
                targetPhi = 0.0;
            } else if (gate === "X") {
                // Pauli-X bit flip
                targetTheta = Math.PI - targetTheta;
                targetPhi = targetPhi + Math.PI;
            } else if (gate === "Y") {
                // Pauli-Y bit & phase flip
                targetTheta = Math.PI - targetTheta;
                targetPhi = targetPhi + Math.PI / 2;
            } else if (gate === "Z") {
                // Pauli-Z phase flip
                targetPhi = targetPhi + Math.PI;
            } else if (gate === "S") {
                // Phase gate: adds pi/2 to phi
                targetPhi = targetPhi + Math.PI / 2;
            } else if (gate === "T") {
                // T-gate: adds pi/4 to phi
                targetPhi = targetPhi + Math.PI / 4;
            } else if (gate === "Rx") {
                // Rotation around X axis
                targetTheta = (targetTheta + Math.PI / 4) % Math.PI;
            } else if (gate === "Ry") {
                // Rotation around Y axis
                targetTheta = (targetTheta + Math.PI / 4) % Math.PI;
            } else if (gate === "Rz") {
                // Rotation around Z axis
                targetPhi = targetPhi + Math.PI / 4;
            } else if (gate === "superposition") {
                targetTheta = Math.PI / 2;
                targetPhi = Math.PI / 4;
            } else if (gate === "reset") {
                targetTheta = 0.01;
                targetPhi = 0.0;
                trail = [];
            }
        }

        window.applyQuantumGate = applyGate;

        // UI Math & Probability Elements
        const p0Elem = document.getElementById("prob0");
        const p1Elem = document.getElementById("prob1");
        const p0Bar = document.getElementById("prob0Bar");
        const p1Bar = document.getElementById("prob1Bar");
        const stateFormulaElem = document.getElementById("blochStateFormula");
        const thetaValElem = document.getElementById("blochThetaVal");
        const phiValElem = document.getElementById("blochPhiVal");

        function updateProbabilities() {
            const p0 = Math.cos(theta / 2) ** 2;
            const p1 = Math.sin(theta / 2) ** 2;
            const alpha = Math.cos(theta / 2);
            const beta = Math.sin(theta / 2);

            if (p0Elem) p0Elem.textContent = (p0 * 100).toFixed(1) + "%";
            if (p1Elem) p1Elem.textContent = (p1 * 100).toFixed(1) + "%";
            if (p0Bar) p0Bar.style.width = (p0 * 100) + "%";
            if (p1Bar) p1Bar.style.width = (p1 * 100) + "%";

            if (stateFormulaElem) {
                const phiDeg = ((phi * 180 / Math.PI) % 360).toFixed(0);
                stateFormulaElem.textContent = `|ψ⟩ = ${alpha.toFixed(3)}|0⟩ + ${beta.toFixed(3)}·e^(${phiDeg}°i)|1⟩`;
            }
            if (thetaValElem) thetaValElem.textContent = `θ = ${(theta * 180 / Math.PI).toFixed(1)}° (${theta.toFixed(2)} rad)`;
            if (phiValElem) phiValElem.textContent = `φ = ${((phi * 180 / Math.PI) % 360).toFixed(1)}° (${(phi % (2 * Math.PI)).toFixed(2)} rad)`;
        }

        function project3D(x, y, z, cx, cy, sphereRadius) {
            const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
            const cosX = Math.cos(rotX), sinX = Math.sin(rotX);

            const x1 = x * cosY - z * sinY;
            const z1 = x * sinY + z * cosY;

            const y2 = y * cosX - z1 * sinX;
            const z2 = y * sinX + z1 * cosX;

            const fov = 380;
            const scale = fov / (z2 + 460);
            return {
                x: cx + x1 * scale * sphereRadius,
                y: cy - y2 * scale * sphereRadius,
                z: z2,
            };
        }

        function render() {
            ctx.clearRect(0, 0, width, height);

            const cx = width / 2;
            const cy = height / 2;
            const sphereRadius = Math.min(width, height) * 0.38;

            // Interpolate towards target angles
            theta += (targetTheta - theta) * 0.08;
            phi += (targetPhi - phi) * 0.08;
            phi += precessionSpeed; // Continuous Larmor precession

            updateProbabilities();

            // 1. Draw Background Outer Sphere Glow
            const radGrad = ctx.createRadialGradient(cx, cy, sphereRadius * 0.2, cx, cy, sphereRadius * 1.05);
            radGrad.addColorStop(0, "rgba(139, 92, 246, 0.04)");
            radGrad.addColorStop(0.8, "rgba(6, 182, 212, 0.03)");
            radGrad.addColorStop(1, "transparent");
            ctx.fillStyle = radGrad;
            ctx.beginPath();
            ctx.arc(cx, cy, sphereRadius, 0, Math.PI * 2);
            ctx.fill();

            // 2. Latitude Circles (Wireframe Grid)
            ctx.strokeStyle = "rgba(51, 65, 85, 0.35)";
            ctx.lineWidth = 1;

            // Equator (z = 0)
            ctx.beginPath();
            for (let a = 0; a <= Math.PI * 2; a += 0.08) {
                const pt = project3D(Math.cos(a), 0, Math.sin(a), cx, cy, sphereRadius);
                if (a === 0) ctx.moveTo(pt.x, pt.y);
                else ctx.lineTo(pt.x, pt.y);
            }
            ctx.stroke();

            // Meridian Circle (x = 0)
            ctx.beginPath();
            for (let a = 0; a <= Math.PI * 2; a += 0.08) {
                const pt = project3D(0, Math.cos(a), Math.sin(a), cx, cy, sphereRadius);
                if (a === 0) ctx.moveTo(pt.x, pt.y);
                else ctx.lineTo(pt.x, pt.y);
            }
            ctx.stroke();

            // Additional Prime Meridian (y = 0)
            ctx.beginPath();
            for (let a = 0; a <= Math.PI * 2; a += 0.08) {
                const pt = project3D(Math.sin(a), Math.cos(a), 0, cx, cy, sphereRadius);
                if (a === 0) ctx.moveTo(pt.x, pt.y);
                else ctx.lineTo(pt.x, pt.y);
            }
            ctx.stroke();

            // 3. Draw Principal Axes (X, Y, Z)
            const axes = [
                { name: "+X |+⟩", color: "#EF4444", vec: [1.25, 0, 0] },
                { name: "-X |-⟩", color: "rgba(239, 68, 68, 0.6)", vec: [-1.25, 0, 0] },
                { name: "+Y |i⟩", color: "#10B981", vec: [0, 0, 1.25] },
                { name: "-Y |-i⟩", color: "rgba(16, 185, 129, 0.6)", vec: [0, 0, -1.25] },
                { name: "+Z |0⟩", color: "#06B6D4", vec: [0, 1.25, 0] },
                { name: "-Z |1⟩", color: "#A855F7", vec: [0, -1.25, 0] },
            ];

            const centerPt = project3D(0, 0, 0, cx, cy, sphereRadius);

            axes.forEach(axis => {
                const endPt = project3D(axis.vec[0], axis.vec[1], axis.vec[2], cx, cy, sphereRadius);
                ctx.beginPath();
                ctx.moveTo(centerPt.x, centerPt.y);
                ctx.lineTo(endPt.x, endPt.y);
                ctx.strokeStyle = axis.color;
                ctx.lineWidth = axis.name.startsWith("+") ? 1.6 : 1.0;
                ctx.stroke();

                ctx.fillStyle = axis.color;
                ctx.font = "bold 11px JetBrains Mono";
                ctx.fillText(axis.name, endPt.x + 5, endPt.y + 4);
            });

            // 4. Calculate Current Statevector Coordinates on Sphere
            const sx = Math.sin(theta) * Math.cos(phi);
            const sz = Math.sin(theta) * Math.sin(phi);
            const sy = Math.cos(theta); // Quantum Z mapped to 3D Y coordinate

            const statePt = project3D(sx, sy, sz, cx, cy, sphereRadius);

            // Record trajectory trail point
            trail.push({ x: sx, y: sy, z: sz });
            if (trail.length > 55) trail.shift();

            // 5. Draw Dynamic Quantum Trajectory Trail
            if (trail.length > 1) {
                for (let i = 1; i < trail.length; i++) {
                    const ptA = project3D(trail[i - 1].x, trail[i - 1].y, trail[i - 1].z, cx, cy, sphereRadius);
                    const ptB = project3D(trail[i].x, trail[i].y, trail[i].z, cx, cy, sphereRadius);
                    const alpha = (i / trail.length) * 0.75;
                    ctx.beginPath();
                    ctx.moveTo(ptA.x, ptA.y);
                    ctx.lineTo(ptB.x, ptB.y);
                    ctx.strokeStyle = `rgba(245, 158, 11, ${alpha})`;
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }
            }

            // 6. Draw Statevector Arrow: |ψ⟩
            ctx.beginPath();
            ctx.moveTo(centerPt.x, centerPt.y);
            ctx.lineTo(statePt.x, statePt.y);
            ctx.strokeStyle = "#F59E0B";
            ctx.lineWidth = 3.5;
            ctx.shadowColor = "#F59E0B";
            ctx.shadowBlur = 15;
            ctx.stroke();
            ctx.shadowBlur = 0;

            // Statevector tip glowing marker
            ctx.beginPath();
            ctx.arc(statePt.x, statePt.y, 7.5, 0, Math.PI * 2);
            ctx.fillStyle = "#F59E0B";
            ctx.shadowColor = "#FCD34D";
            ctx.shadowBlur = 18;
            ctx.fill();
            ctx.shadowBlur = 0;

            // Label
            ctx.fillStyle = "#FCD34D";
            ctx.font = "bold 13px JetBrains Mono";
            ctx.fillText("|ψ⟩", statePt.x + 9, statePt.y - 6);

            requestAnimationFrame(render);
        }

        render();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initBlochSphere);
    } else {
        initBlochSphere();
    }
})();
