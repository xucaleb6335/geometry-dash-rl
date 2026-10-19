architecture.mjs draws docs/media/architecture.png in an Excalidraw-like hand-drawn style
(rough.js shapes + the Virgil font, both from the Excalidraw ecosystem).
  npm i roughjs@4.6.6
  curl -L -o Virgil.woff2 https://unpkg.com/@excalidraw/excalidraw@0.17.6/dist/excalidraw-assets/Virgil.woff2
  node architecture.mjs        # writes architecture.svg / architecture.html
  chrome --headless=new --force-device-scale-factor=2 --window-size=1640,830 --screenshot=architecture.png architecture.html
