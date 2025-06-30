`timescale 1ns / 1ps

module top (
    input  wire                               clk,
    input  wire                               rst_n,
    input  wire                               sclk,
    input  wire                               mosi,
    input  wire                               cs,
    output wire                               o_valid,
    input  wire                               o_ready,
    output wire [1:0]                         o_predicted_class
);


    localparam PARALLEL_FACTOR   = 8;
    localparam HIDDEN_DATA_WIDTH = 8;
    localparam LOGIT_WIDTH       = 16;
    localparam ACC_WIDTH         = 32;
    localparam BIAS_WIDTH        = 32;

    localparam P1_K_FEATURES   = 300;
    localparam P2_OUTPUT_SIZE  = 256;
    localparam P3_OUTPUT_SIZE  = 128;
    localparam P4_OUTPUT_SIZE  = 3;

    // 量化参数...
    localparam signed [7:0]  LAYER1_IN_ZP    = 0;
    localparam signed [8:0]  LAYER1_OUT_ZP   = 180;
    localparam signed [31:0] LAYER1_OUT_MULT = 1198999149;
    localparam integer       LAYER1_OUT_SHIFT  = 11;
    localparam signed [8:0]  LAYER2_IN_ZP    = LAYER1_OUT_ZP;
    localparam signed [8:0]  LAYER2_OUT_ZP   = 104;
    localparam signed [31:0] LAYER2_OUT_MULT = 1672348687;
    localparam integer       LAYER2_OUT_SHIFT  = 8;
    localparam signed [8:0]  LAYER3_IN_ZP    = LAYER2_OUT_ZP;
    localparam integer       LAYER3_LOGITS_SHIFT = 8;
    
    // ROM 
    localparam P2_WEIGHT_ROM_FILE = "layer_0_weights_p8.mem";
    localparam P2_BIAS_ROM_FILE   = "layer_0_bias.mem";
    localparam P3_WEIGHT_ROM_FILE = "layer_1_weights_p8.mem";
    localparam P3_BIAS_ROM_FILE   = "layer_1_bias.mem";
    localparam P4_WEIGHT_ROM_FILE = "layer_2_weights_p8.mem";
    localparam P4_BIAS_ROM_FILE   = "layer_2_bias.mem";

    wire                          spi_to_top_valid;
    wire                          top_to_spi_ready;
    wire signed [P1_K_FEATURES*HIDDEN_DATA_WIDTH-1:0] feature_data_bus;

    wire signed [P2_OUTPUT_SIZE*HIDDEN_DATA_WIDTH-1:0] w_p2_spikes;
    wire signed [P3_OUTPUT_SIZE*HIDDEN_DATA_WIDTH-1:0] w_p3_spikes;
    wire signed [P4_OUTPUT_SIZE*LOGIT_WIDTH-1:0]       w_p4_logits;
    wire w_valid_p2_to_p3, w_ready_p3_to_p2;
    wire w_valid_p3_to_p4, w_ready_p4_to_p3;
    wire w_valid_p4_to_p5, w_ready_p5_to_p4;


    // PI---
    spi_slave_rx #(
        .DATA_WIDTH      (HIDDEN_DATA_WIDTH),
        .FEATURE_LENGTH  (P1_K_FEATURES)
    ) spi_rx_inst (
        .clk          (clk),
        .rst_n        (rst_n),
        .sclk         (sclk),
        .mosi         (mosi),
        .cs           (cs),
        .o_valid      (spi_to_top_valid),
        .i_ready      (top_to_spi_ready),
        .o_features   (feature_data_bus)
    );

    //  1: dense_layer_1
    dense_layer #(
        .PARALLEL_FACTOR      (PARALLEL_FACTOR), 
        .INPUT_SIZE           (P1_K_FEATURES),
        .OUTPUT_SIZE          (P2_OUTPUT_SIZE),
        .DATA_WIDTH           (HIDDEN_DATA_WIDTH),
        .ACC_WIDTH            (ACC_WIDTH),
        .BIAS_WIDTH           (BIAS_WIDTH),
        .WEIGHT_ROM_FILE      (P2_WEIGHT_ROM_FILE),
        .BIAS_ROM_FILE        (P2_BIAS_ROM_FILE),
        .INPUT_ZERO_POINT     (LAYER1_IN_ZP),
        .OUTPUT_ZERO_POINT    (LAYER1_OUT_ZP),
        .OUTPUT_MULTIPLIER    (LAYER1_OUT_MULT),
        .OUTPUT_SHIFT         (LAYER1_OUT_SHIFT)
    ) dense_layer_1_inst (
        .clk(clk), 
        .rst_n(rst_n), 
        .i_valid(spi_to_top_valid),
        .i_ready(top_to_spi_ready),
        .o_valid(w_valid_p2_to_p3), 
        .o_ready(w_ready_p3_to_p2), 
        .i_spikes(feature_data_bus),
        .o_spikes(w_p2_spikes)
    );

    //  dense_layer_2 
    dense_layer #(
        .PARALLEL_FACTOR      (PARALLEL_FACTOR), 
        .INPUT_SIZE           (P2_OUTPUT_SIZE),
        .OUTPUT_SIZE          (P3_OUTPUT_SIZE),
        .DATA_WIDTH           (HIDDEN_DATA_WIDTH),
        .ACC_WIDTH            (ACC_WIDTH),
        .BIAS_WIDTH           (BIAS_WIDTH),
        .WEIGHT_ROM_FILE      (P3_WEIGHT_ROM_FILE),
        .BIAS_ROM_FILE        (P3_BIAS_ROM_FILE),
        .INPUT_ZERO_POINT     (LAYER2_IN_ZP),
        .OUTPUT_ZERO_POINT    (LAYER2_OUT_ZP),
        .OUTPUT_MULTIPLIER    (LAYER2_OUT_MULT),
        .OUTPUT_SHIFT         (LAYER2_OUT_SHIFT)
    ) dense_layer_2_inst ( 
        .clk(clk), .rst_n(rst_n), .i_valid(w_valid_p2_to_p3), .i_ready(w_ready_p3_to_p2), .o_valid(w_valid_p3_to_p4), .o_ready(w_ready_p4_to_p3), .i_spikes(w_p2_spikes), .o_spikes(w_p3_spikes)
    );

    // --- output_layer ---
    output_layer #(
        .PARALLEL_FACTOR      (PARALLEL_FACTOR),      
        .INPUT_SIZE           (P3_OUTPUT_SIZE),
        .OUTPUT_SIZE          (P4_OUTPUT_SIZE),
        .INPUT_DATA_WIDTH     (HIDDEN_DATA_WIDTH),
        .OUTPUT_DATA_WIDTH    (LOGIT_WIDTH),
        .ACC_WIDTH            (ACC_WIDTH),
        .BIAS_WIDTH           (BIAS_WIDTH),
        .WEIGHT_ROM_FILE      (P4_WEIGHT_ROM_FILE),
        .BIAS_ROM_FILE        (P4_BIAS_ROM_FILE),
        .LOGITS_SCALE_SHIFT   (LAYER3_LOGITS_SHIFT),
        .INPUT_ZERO_POINT     (LAYER3_IN_ZP)        
    ) output_layer_inst ( 
        .clk(clk), .rst_n(rst_n), .i_valid(w_valid_p3_to_p4), .i_ready(w_ready_p4_to_p3), .o_valid(w_valid_p4_to_p5), .o_ready(w_ready_p5_to_p4), .i_spikes(w_p3_spikes), .o_logits(w_p4_logits)
    );

    // --- 4: Argmax ---
    argmax #(
        .DATA_WIDTH           (LOGIT_WIDTH)
    ) argmax_inst (
        .clk(clk), .rst_n(rst_n), .i_valid(w_valid_p4_to_p5), .i_ready(w_ready_p5_to_p4), .o_ready(o_ready), .o_valid(o_valid), .i_logits(w_p4_logits), .o_predicted_class(o_predicted_class)
    );

endmodule