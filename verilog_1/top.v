`timescale 1ns / 1ps

module top (
    // =================================================================
    // == 顶层端口定义
    // =================================================================
    // --- 系统信号 ---
    input  wire                          clk,
    input  wire                          rst_n,
    // --- 顶层握手接口 ---
    input  wire                          i_valid,             // 外部系统发出有效数据
    output wire                          i_ready,             // 加速器准备好接收新数据
    output wire                          o_valid,             // 加速器输出有效结果
    input  wire                          o_ready,             // 外部系统准备好接收结果
    // --- 顶层数据接口 (已修改) ---
    input  wire signed [300*32-1:0]      i_features,          // 修改: 输入特征向量直接为300个
    output wire [1:0]                    o_predicted_class    // 最终输出的类别索引 (0, 1, 或 2)
);

    // =================================================================
    // == 内部参数定义
    // =================================================================
    localparam DATA_WIDTH      = 32;
    localparam FRACTIONAL_BITS = 16;

    // --- 各层尺寸定义 (已更新) ---
    // localparam P0_TOTAL_FEATURES = 682; // 已移除
    localparam P1_K_FEATURES   = 300; // 现在是输入层
    localparam P2_OUTPUT_SIZE  = 256; // 第1隐藏层后
    localparam P3_OUTPUT_SIZE  = 128; // 第2隐藏层后
    localparam P4_OUTPUT_SIZE  = 3;   // 输出层后

    // --- ROM 文件路径定义 (已更新) ---
    localparam P2_WEIGHT_ROM_FILE = "layer_0_weights.mem";
    localparam P2_BIAS_ROM_FILE   = "layer_0_combined_bias.mem";
    localparam P3_WEIGHT_ROM_FILE = "layer_1_weights.mem";
    localparam P3_BIAS_ROM_FILE   = "layer_1_combined_bias.mem";
    localparam P4_WEIGHT_ROM_FILE = "layer_2_weights.mem";
    localparam P4_BIAS_ROM_FILE   = "layer_2_bias.mem";



    // --- 流水线数据连线 ---
    wire signed [P2_OUTPUT_SIZE*DATA_WIDTH-1:0] w_p2_spikes;    // P2 -> P3
    wire signed [P3_OUTPUT_SIZE*DATA_WIDTH-1:0] w_p3_spikes;    // P3 -> P4
    wire signed [P4_OUTPUT_SIZE*DATA_WIDTH-1:0] w_p4_logits;    // P4 -> P5

    // --- 流水线握手信号连线 ---
    wire w_valid_p2_to_p3, w_ready_p3_to_p2;
    wire w_valid_p3_to_p4, w_ready_p4_to_p3;
    wire w_valid_p4_to_p5, w_ready_p5_to_p4;

    // --- 阶段1: 第一个隐藏层 ---
    dense_layer #(
        .INPUT_SIZE           (P1_K_FEATURES),
        .OUTPUT_SIZE          (P2_OUTPUT_SIZE),
        .DATA_WIDTH           (DATA_WIDTH),
        .FRACTIONAL_BITS      (FRACTIONAL_BITS),
        .WEIGHT_ROM_FILE      (P2_WEIGHT_ROM_FILE),
        .BIAS_ROM_FILE        (P2_BIAS_ROM_FILE),
        .T_MIN                (32'h00010000), 
        .T_MAX                (32'h00020000)
    ) dense_layer_1_inst (
        .clk                  (clk),
        .rst_n                (rst_n),
        // 修改: 直接连接到顶层输入
        .i_valid              (i_valid),
        .i_ready              (i_ready),
        .o_valid              (w_valid_p2_to_p3),
        .o_ready              (w_ready_p3_to_p2),
        // 修改: 直接使用顶层输入i_features
        .i_spikes             (i_features),
        .o_spikes             (w_p2_spikes)
    );

    // --- 阶段2: 第二个隐藏层  ---
    dense_layer #(
        .INPUT_SIZE           (P2_OUTPUT_SIZE),
        .OUTPUT_SIZE          (P3_OUTPUT_SIZE),
        .DATA_WIDTH           (DATA_WIDTH),
        .FRACTIONAL_BITS      (FRACTIONAL_BITS),
        .WEIGHT_ROM_FILE      (P3_WEIGHT_ROM_FILE),
        .BIAS_ROM_FILE        (P3_BIAS_ROM_FILE),
        .T_MIN                (32'h00020000), 
        .T_MAX                (32'h00030000)
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

    // --- 阶段3: 输出层 ---
    output_layer #(
        .INPUT_SIZE           (P3_OUTPUT_SIZE),
        .OUTPUT_SIZE          (P4_OUTPUT_SIZE),
        .DATA_WIDTH           (DATA_WIDTH),
        .FRACTIONAL_BITS      (FRACTIONAL_BITS),
        .WEIGHT_ROM_FILE      (P4_WEIGHT_ROM_FILE),
        .BIAS_ROM_FILE        (P4_BIAS_ROM_FILE)
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

    // --- 阶段4: Argmax决策 ---
    argmax #(
        .DATA_WIDTH (DATA_WIDTH)
    ) argmax_inst (
        .clk               (clk),
        .rst_n             (rst_n),
        .i_valid           (w_valid_p4_to_p5),
        .i_ready           (w_ready_p5_to_p4),
        .o_ready           (o_ready),
        .o_valid           (o_valid),
        .i_logits          (w_p4_logits),
        .o_predicted_class (o_predicted_class)
    );

endmodule