#include <libobsensor/ObSensor.hpp>
#include <iostream>

int main() {
    try {
        ob::Pipeline pipe;
        auto config = std::make_shared<ob::Config>();
        config->enableVideoStream(OB_STREAM_DEPTH, 640, 400, 15, OB_FORMAT_RLE);

        std::cout << "Starting depth pipeline: 640x400 @ 15 FPS, RLE..." << std::endl;
        pipe.start(config);
        std::cout << "Pipeline started." << std::endl;

        int framesets = 0;
        int depth_frames = 0;

        for(int i = 0; i < 100 && depth_frames < 10; ++i) {
            auto frameset = pipe.waitForFrameset(200);
            if(!frameset) {
                continue;
            }
            ++framesets;

            auto raw = frameset->getFrame(OB_FRAME_DEPTH);
            if(!raw) {
                continue;
            }

            auto depth = raw->as<ob::DepthFrame>();
            ++depth_frames;
            std::cout << "Depth frame " << depth_frames
                      << " | " << depth->getWidth() << "x" << depth->getHeight()
                      << " | format=" << depth->getFormat()
                      << " | scale=" << depth->getValueScale()
                      << std::endl;
        }

        pipe.stop();
        std::cout << "RESULT: framesets=" << framesets
                  << ", depth_frames=" << depth_frames << std::endl;
        return depth_frames > 0 ? 0 : 2;
    }
    catch(const ob::Error &e) {
        std::cerr << "Orbbec error\n"
                  << "function: " << e.getFunction() << "\n"
                  << "args: " << e.getArgs() << "\n"
                  << "message: " << e.what() << "\n"
                  << "status: " << e.getStatus() << "\n"
                  << "type: " << e.getExceptionType() << std::endl;
        return 1;
    }
}
