from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parents[4]

SIGNATURE_DIR = (
    BASE_DIR
    / "data"
    / "signature_classifier"
    / "train"
    / "signature"
)

SIGNATURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


app = FastAPI(
    title="Signature Dataset Collector",
)


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Signature Collector</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            text-align: center;
        }

        canvas {
            border: 2px solid #222;
            background: white;
            cursor: crosshair;
            touch-action: none;
        }

        button {
            margin: 15px 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }

        #status {
            margin-top: 15px;
            font-weight: bold;
        }
    </style>
</head>

<body>

<h1>Signature Dataset Collector</h1>

<p>
Draw your signature inside the box.
</p>

<canvas
    id="canvas"
    width="800"
    height="300">
</canvas>

<br>

<button onclick="clearCanvas()">
    Clear
</button>

<button onclick="saveSignature()">
    Save Signature
</button>

<div id="status"></div>

<script>

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

ctx.fillStyle = "white";
ctx.fillRect(
    0,
    0,
    canvas.width,
    canvas.height
);

ctx.strokeStyle = "black";
ctx.lineWidth = 3;
ctx.lineCap = "round";
ctx.lineJoin = "round";

let drawing = false;


function getPosition(event) {

    const rect = canvas.getBoundingClientRect();

    return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
    };
}


canvas.addEventListener(
    "pointerdown",
    function(event) {

        drawing = true;

        const position =
            getPosition(event);

        ctx.beginPath();

        ctx.moveTo(
            position.x,
            position.y
        );
    }
);


canvas.addEventListener(
    "pointermove",
    function(event) {

        if (!drawing) {
            return;
        }

        const position =
            getPosition(event);

        ctx.lineTo(
            position.x,
            position.y
        );

        ctx.stroke();
    }
);


canvas.addEventListener(
    "pointerup",
    function() {

        drawing = false;
        ctx.closePath();
    }
);


canvas.addEventListener(
    "pointerleave",
    function() {

        drawing = false;
    }
);


function clearCanvas() {

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    ctx.fillStyle = "white";

    ctx.fillRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    ctx.strokeStyle = "black";
    ctx.lineWidth = 3;

    document.getElementById(
        "status"
    ).innerText = "";
}


async function saveSignature() {

    const image =
        canvas.toDataURL(
            "image/png"
        );

    const response =
        await fetch(
            "/save",
            {
                method: "POST",
                headers: {
                    "Content-Type":
                        "application/json"
                },
                body: JSON.stringify({
                    image: image
                })
            }
        );

    const result =
        await response.json();

    document.getElementById(
        "status"
    ).innerText =
        result.message;

    clearCanvas();
}

</script>

</body>
</html>
"""


@app.get(
    "/",
    response_class=HTMLResponse,
)
def collector():

    return HTML


@app.post("/save")
def save_signature(payload: dict):

    image_data = payload.get(
        "image"
    )

    if not image_data:

        return {
            "success": False,
            "message": "No signature received.",
        }

    try:

        # Remove data URL prefix.
        encoded = image_data.split(
            ",",
            1
        )[1]

        import base64

        image_bytes = base64.b64decode(
            encoded
        )

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:

            return {
                "success": False,
                "message": "Invalid image.",
            }

        # -------------------------------------------------
        # Reject blank canvas
        # -------------------------------------------------

        _, threshold = cv2.threshold(
            image,
            245,
            255,
            cv2.THRESH_BINARY_INV,
        )

        if cv2.countNonZero(
            threshold
        ) < 50:

            return {
                "success": False,
                "message": (
                    "Signature is empty. "
                    "Please draw first."
                ),
            }

        # -------------------------------------------------
        # Generate sequential filename
        # -------------------------------------------------

        existing = list(
            SIGNATURE_DIR.glob(
                "signature_*.png"
            )
        )

        number = (
            len(existing) + 1
        )

        output_path = (
            SIGNATURE_DIR
            / f"signature_{number:04d}.png"
        )

        cv2.imwrite(
            str(output_path),
            image,
        )

        return {
            "success": True,
            "message": (
                f"Saved signature_{number:04d}.png"
            ),
        }

    except Exception as exc:

        return {
            "success": False,
            "message": (
                f"Failed to save signature: {exc}"
            ),
        }