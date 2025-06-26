`timescale 1ns / 1ps

module top (
    // --- 系统信号 ---
    input  wire                          clk,
    input  wire                          rst_n,
    // --- 顶层握手接口 ---
    input  wire                          i_valid,
    output wire                          i_ready,
    output wire                          o_valid,
    input  wire                          o_ready,
    // --- 顶层数据接口 ---
    input  wire signed [300*8-1:0]       i_features,
    output wire [1:0]                    o_predicted_class
);

    // =================================================================
    // == 内部参数定义
    // =================================================================
    // [修改] 为隐藏层和输出层定义不同的数据位宽
    localparam HIDDEN_DATA_WIDTH = 8;    // 隐藏层之间的数据位宽 (INT8)
    localparam LOGIT_WIDTH       = 16;   // 输出层Logits的数据位宽 (INT16)

    localparam ACC_WIDTH         = 32;
    localparam BIAS_WIDTH        = 32;

    // --- 各层尺寸定义 ---
    localparam P1_K_FEATURES   = 300;
    localparam P2_OUTPUT_SIZE  = 256;
    localparam P3_OUTPUT_SIZE  = 128;
    localparam P4_OUTPUT_SIZE  = 3;

   // --- INT8量化参数定义 ---

    // 第1个隐藏层 (dense_layer_1) 的量化参数
    localparam signed [7:0]  LAYER1_IN_ZP    = 0;          // 来自 layer_0.input_zero_point
    localparam signed [8:0]  LAYER1_OUT_ZP   = 180;        // 来自 layer_1.input_zero_point
    localparam signed [31:0] LAYER1_OUT_MULT = 1198999149; // 来自 layer_0.requant_M_int32
    localparam integer       LAYER1_OUT_SHIFT  = 11;         // 来自 layer_0.requant_N_int

    // 第2个隐藏层 (dense_layer_2) 的量化参数
    localparam signed [8:0]  LAYER2_IN_ZP    = LAYER1_OUT_ZP; // 第2层的输入零点 = 第1层的输出零点 (180)
    localparam signed [8:0]  LAYER2_OUT_ZP   = 104;        // 来自 layer_2.input_zero_point
    localparam signed [31:0] LAYER2_OUT_MULT = 1672348687; // 来自 layer_1.requant_M_int32
    localparam integer       LAYER2_OUT_SHIFT  = 8;          // 来自 layer_1.requant_N_int

    // 输出层 (output_layer) 的参数
    localparam signed [8:0]  LAYER3_IN_ZP    = LAYER2_OUT_ZP; // 输出层的输入零点 = 第2层的输出零点 (104)
    
    // !!! 重要: 这个值需要通过仿真调试来最终确定 !!!
    localparam integer       LAYER3_LOGITS_SHIFT = 8;

    // --- ROM 文件路径定义 ---
    localparam P2_WEIGHT_ROM_FILE = "layer_0_weights.mem";
    localparam P2_BIAS_ROM_FILE   = "layer_0_bias.mem";
    localparam P3_WEIGHT_ROM_FILE = "layer_1_weights.mem";
    localparam P3_BIAS_ROM_FILE   = "layer_1_bias.mem";
    localparam P4_WEIGHT_ROM_FILE = "layer_2_weights.mem";
    localparam P4_BIAS_ROM_FILE   = "layer_2_bias.mem";

    // --- [修改] 流水线数据连线位宽 ---
    wire signed [P2_OUTPUT_SIZE*HIDDEN_DATA_WIDTH-1:0] w_p2_spikes; // 256 * 8
    wire signed [P3_OUTPUT_SIZE*HIDDEN_DATA_WIDTH-1:0] w_p3_spikes; // 128 * 8
    wire signed [P4_OUTPUT_SIZE*LOGIT_WIDTH-1:0]       w_p4_logits; // 3 * 16 

    // --- 流水线握手信号连线 ---
    wire w_valid_p2_to_p3, w_ready_p3_to_p2;
    wire w_valid_p3_to_p4, w_ready_p4_to_p3;
    wire w_valid_p4_to_p5, w_ready_p5_to_p4;

    // --- 阶段1: 第一个隐藏层 ---
    dense_layer #(
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
        .clk                  (clk),
        .rst_n                (rst_n),
        .i_valid              (i_valid),
        .i_ready              (i_ready),
        .o_valid              (w_valid_p2_to_p3),
        .o_ready              (w_ready_p3_to_p2),
        .i_spikes             (i_features),
        .o_spikes             (w_p2_spikes)
    );

    // --- 阶段2: 第二个隐藏层 ---
    dense_layer #(
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
        .clk                  (clk),
        .rst_n                (rst_n),
        .i_valid              (w_valid_p2_to_p3),
        .i_ready              (w_ready_p3_to_p2),
        .o_valid              (w_valid_p3_to_p4),
        .o_ready              (w_ready_p4_to_p3),
        .i_spikes             (w_p2_spikes),
        .o_spikes             (w_p3_spikes)
    );

    // --- [修改] 阶段3: 输出层 ---
    // 实例化修改后的 output_layer 模块
    output_layer #(
        .INPUT_SIZE           (P3_OUTPUT_SIZE),
        .OUTPUT_SIZE          (P4_OUTPUT_SIZE),
        .INPUT_DATA_WIDTH     (HIDDEN_DATA_WIDTH), // 输入仍然是8位
        .OUTPUT_DATA_WIDTH    (LOGIT_WIDTH),       // 输出是16位
        .ACC_WIDTH            (ACC_WIDTH),
        .BIAS_WIDTH           (BIAS_WIDTH),
        .WEIGHT_ROM_FILE      (P4_WEIGHT_ROM_FILE),
        .BIAS_ROM_FILE        (P4_BIAS_ROM_FILE),
        .LOGITS_SCALE_SHIFT   (LAYER3_LOGITS_SHIFT)
    ) output_layer_inst (
        .clk                  (clk),
        .rst_n                (rst_n),
        .i_valid              (w_valid_p3_to_p4),
        .i_ready              (w_ready_p4_to_p3),
        .o_valid              (w_valid_p4_to_p5),
        .o_ready              (w_ready_p5_to_p4),
        .i_spikes             (w_p3_spikes),
        .o_logits             (w_p4_logits)
    );

    // --- [修改] 阶段4: Argmax决策 ---
    // Argmax模块的数据位宽现在由LOGIT_WIDTH参数化
    argmax #(
        .DATA_WIDTH           (LOGIT_WIDTH) // 传入16位
    ) argmax_inst (
        .clk                  (clk),
        .rst_n                (rst_n),
        .i_valid              (w_valid_p4_to_p5),
        .i_ready              (w_ready_p5_to_p4),
        .o_ready              (o_ready),
        .o_valid              (o_valid),
        .i_logits             (w_p4_logits),
        .o_predicted_class    (o_predicted_class)
    );
endmodule