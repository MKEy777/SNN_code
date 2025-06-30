// 文件名: quantization_unit_3stage_final.v
// 模块: quantization_unit (3级流水线最终版)
// 修正: 深化为3级流水线，将加法和乘法分离，以获得最大的时序裕量。
`timescale 1ns / 1ps

module quantization_unit #(
    parameter integer DATA_WIDTH       = 8,
    parameter integer ACC_WIDTH        = 32,
    parameter integer BIAS_WIDTH       = 32,
    parameter signed [31:0] OUTPUT_ZERO_POINT  = 180,
    parameter signed [31:0] OUTPUT_MULTIPLIER  = 1198999149,
    parameter integer       OUTPUT_SHIFT       = 11
)(
    input  wire                                clk,
    input  wire                                rst_n,
    input  wire                                i_valid,
    output wire                                o_valid,
    input  wire signed [ACC_WIDTH-1:0]       accumulator_in,
    input  wire signed [BIAS_WIDTH-1:0]        bias_in,
    output wire signed [DATA_WIDTH-1:0]       clamped_output
);

    // =================================================================
    // == 1. 流水线寄存器定义
    // =================================================================
    // --- 级1 -> 级2 数据通路寄存器: 存储加法结果 ---
    reg signed [ACC_WIDTH:0]    acc_with_bias_reg;

    // --- 级2 -> 级3 数据通路寄存器: 存储乘法结果 ---
    reg signed [ACC_WIDTH+32:0] mult_reg; // 位宽修正以匹配乘法器输出

    // --- 控制通路寄存器 (3级流水线 -> 3周期延迟) ---
    reg [2:0] valid_pipeline_reg;
    assign o_valid = valid_pipeline_reg[2];

    // =================================================================
    // == 2. 组合逻辑定义
    // =================================================================
    // --- 级1 的计算逻辑 (加法) ---
    wire signed [ACC_WIDTH:0] acc_with_bias_comb = accumulator_in + bias_in;

    // --- 级2 的计算逻辑 (乘法) ---
    wire signed [ACC_WIDTH+32:0] mult_comb = acc_with_bias_reg * OUTPUT_MULTIPLIER;

    // --- 级3 的计算逻辑 (移位, 加零点, 饱和) ---
    wire signed [ACC_WIDTH+32:0] shifted = $signed(mult_reg) >>> OUTPUT_SHIFT;
    wire signed [ACC_WIDTH+32:0] with_zp;
    wire signed [ACC_WIDTH+32:0] output_zero_point_extended = OUTPUT_ZERO_POINT;
    assign with_zp = shifted + output_zero_point_extended;
    
    localparam signed [DATA_WIDTH-1:0] MAX_VAL = {1'b0, {DATA_WIDTH-1{1'b1}}};
    localparam signed [DATA_WIDTH-1:0] MIN_VAL = {1'b1, {DATA_WIDTH-1{1'b0}}};
    wire is_overflow  = (with_zp > MAX_VAL);
    wire is_underflow = (with_zp < MIN_VAL);
    assign clamped_output = is_overflow  ? MAX_VAL :
                            is_underflow ? MIN_VAL : with_zp[DATA_WIDTH-1:0];

    // =================================================================
    // == 3. 时序逻辑 (驱动所有流水线寄存器)
    // =================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_with_bias_reg  <= 0;
            mult_reg           <= 0;
            valid_pipeline_reg <= 3'b0;
        end else begin
            // --- 数据流水线 ---
            // 级1 -> 级2: 锁存加法结果
            acc_with_bias_reg <= acc_with_bias_comb;
            
            // 级2 -> 级3: 锁存乘法结果
            mult_reg <= mult_comb;
            
            // --- 控制流水线 ---
            valid_pipeline_reg <= {valid_pipeline_reg[1:0], i_valid};
        end
    end

endmodule