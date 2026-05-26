#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

static std::vector<char> read_binary_file(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    if (!ifs) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::streamsize size = ifs.tellg();
    if (size < 0) {
        throw std::runtime_error("Failed to get file size: " + filename);
    }

    ifs.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<size_t>(size));
    if (!ifs.read(buffer.data(), size)) {
        throw std::runtime_error("Failed to read file: " + filename);
    }
    return buffer;
}

static double mean_ms(const std::vector<double>& values) {
    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
}

static double std_ms(const std::vector<double>& values, double mean) {
    if (values.empty()) {
        return 0.0;
    }
    double acc = 0.0;
    for (double v : values) {
        const double diff = v - mean;
        acc += diff * diff;
    }
    return std::sqrt(acc / static_cast<double>(values.size()));
}

int main(int argc, char** argv) {
    try {
        const std::string xclbin_path = "../bitfile/finn-accel.xclbin";
        const std::string input_path = "input.bin";
        const std::string output_path = "output.bin";
        const std::string timing_path = "board_inference_timing.txt";

        const size_t input_bytes = 1 * 256 * 1;
        const size_t output_elems = 1 * 256;
        const size_t bytes_per_output_elem = 2;
        const size_t output_bytes = output_elems * bytes_per_output_elem;
        const uint32_t num_reps = 1;

        const int measure_runs = (argc > 1) ? std::max(1, std::stoi(argv[1])) : 20;
        const int warmup_runs = (argc > 2) ? std::max(0, std::stoi(argv[2])) : 2;

        std::cout << "Opening device 0..." << std::endl;
        auto device = xrt::device(0);

        std::cout << "Loading xclbin: " << xclbin_path << std::endl;
        auto uuid = device.load_xclbin(xclbin_path);

        std::cout << "Opening kernels..." << std::endl;
        auto idma = xrt::kernel(device, uuid, "StreamingDataflowPartition_0");
        auto odma = xrt::kernel(device, uuid, "StreamingDataflowPartition_2");

        std::cout << "Allocating buffers..." << std::endl;
        auto in_bo = xrt::bo(device, input_bytes, idma.group_id(0));
        auto out_bo = xrt::bo(device, output_bytes, odma.group_id(0));

        auto in_map = in_bo.map<uint8_t*>();
        auto out_map = out_bo.map<uint8_t*>();

        std::cout << "Reading input.bin..." << std::endl;
        auto input_data = read_binary_file(input_path);
        if (input_data.size() != input_bytes) {
            std::cerr << "Input size mismatch. Expected "
                      << input_bytes
                      << " bytes, got "
                      << input_data.size()
                      << " bytes."
                      << std::endl;
            return 1;
        }

        std::vector<double> latency_ms;
        latency_ms.reserve(static_cast<size_t>(measure_runs));

        std::cout << "Warmup runs : " << warmup_runs << std::endl;
        std::cout << "Measure runs: " << measure_runs << std::endl;

        for (int run_idx = 0; run_idx < warmup_runs + measure_runs; ++run_idx) {
            std::memcpy(in_map, input_data.data(), input_bytes);
            std::memset(out_map, 0, output_bytes);

            const auto t0 = std::chrono::high_resolution_clock::now();

            in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            auto odma_run = odma(out_bo, num_reps);
            auto idma_run = idma(in_bo, num_reps);

            idma_run.wait();
            odma_run.wait();
            out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

            const auto t1 = std::chrono::high_resolution_clock::now();
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (run_idx >= warmup_runs) {
                latency_ms.push_back(elapsed_ms);
            }
        }

        std::ofstream ofs(output_path, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error("Cannot open output file: " + output_path);
        }
        ofs.write(
            reinterpret_cast<char*>(out_map),
            static_cast<std::streamsize>(output_bytes)
        );
        ofs.close();

        const double avg = mean_ms(latency_ms);
        const double stddev = std_ms(latency_ms, avg);
        const double min_v = *std::min_element(latency_ms.begin(), latency_ms.end());
        const double max_v = *std::max_element(latency_ms.begin(), latency_ms.end());

        std::ofstream tfs(timing_path);
        if (!tfs) {
            throw std::runtime_error("Cannot open timing output file: " + timing_path);
        }
        tfs << std::fixed << std::setprecision(6);
        tfs << "warmup_runs=" << warmup_runs << "\n";
        tfs << "measure_runs=" << measure_runs << "\n";
        tfs << "mean_ms=" << avg << "\n";
        tfs << "std_ms=" << stddev << "\n";
        tfs << "min_ms=" << min_v << "\n";
        tfs << "max_ms=" << max_v << "\n";
        tfs.close();

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n========== Board Inference Timing ==========" << std::endl;
        std::cout << "Task         : one end-to-end accelerator run" << std::endl;
        std::cout << "Includes     : host->device sync + DMA + accelerator + device->host sync" << std::endl;
        std::cout << "Excludes     : xclbin load, kernel open, buffer allocation" << std::endl;
        std::cout << "Warmup runs  : " << warmup_runs << std::endl;
        std::cout << "Measure runs : " << measure_runs << std::endl;
        std::cout << "Mean latency : " << avg << " ms" << std::endl;
        std::cout << "Std latency  : " << stddev << " ms" << std::endl;
        std::cout << "Min latency  : " << min_v << " ms" << std::endl;
        std::cout << "Max latency  : " << max_v << " ms" << std::endl;
        std::cout << "Saved timing : " << timing_path << std::endl;
        std::cout << "============================================" << std::endl;

        std::cout << "Done. Output saved to " << output_path << std::endl;
        std::cout << "Expected input bytes: " << input_bytes << std::endl;
        std::cout << "Expected output tensor shape: (1, 256)" << std::endl;
        std::cout << "Expected output layout: first 128 real, last 128 imag" << std::endl;
        std::cout << "Expected output datatype: INT16" << std::endl;
        std::cout << "Expected output bytes: " << output_bytes << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
