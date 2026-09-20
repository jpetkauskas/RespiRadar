// RespiRadar on the UNO Q: the STM32 half.
//
// Linux does all the radar work and decides what should be on the screen. This side owns the
// LED matrix and does nothing else: it holds one 8x13 frame and keeps redrawing it until
// Python sends a new one over the Router Bridge.
//
// Keeping the frame here rather than redrawing on receipt matters. The matrix is multiplexed,
// so it has to be driven continuously; if the draw only happened when a frame arrived, the
// display would go dark between updates and any hiccup in the radar loop would look like the
// alarm had stopped.

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <vector>

Arduino_LED_Matrix matrix;

const uint8_t FRAME_ROWS = 8;
const uint8_t FRAME_COLS = 13;
const uint8_t FRAME_SIZE = FRAME_ROWS * FRAME_COLS;

uint8_t frame[FRAME_SIZE] = {0};  // starts dark: nothing is claimed until Linux says so

void setup() {
    matrix.begin();

    // 3 bits, so each LED takes a brightness of 0-7. `respiradar/ledmatrix.py` renders to
    // exactly that range (LEVELS = 8), and `unoq.BridgeSink` is careful to pass those values
    // through unscaled. Raising this to 8 bits here without changing both would turn every
    // lit pixel into a dim flicker.
    matrix.setGrayscaleBits(3);
    matrix.clear();

    Bridge.begin();
    Bridge.provide("draw", draw);
}

void loop() {
    matrix.draw(frame);
    delay(10);
}

// Called from Python with 104 bytes, row-major, one brightness per LED.
void draw(std::vector<uint8_t> newFrame) {
    size_t len = min(newFrame.size(), (size_t)FRAME_SIZE);
    memcpy(frame, newFrame.data(), len);
}
