// 文件名: parallel_compute_engine_pipelined_v2001.v
// 模块: parallel_compute_engine (3级流水线, 严格Verilog-2001版)
// 功能: 并行计算引擎，内部采用3级流水线，以实现更高的时钟频率。
`timescale 1ns / 1ps

module parallel_compute_engine #(
    parameter PARALLEL_FACTOR = 8,
    parameter DATA_WIDTH      = 8,
    parameter ACC_WIDTH       = 32,
    parameter INPUT_ZERO_POINT = 0
)(
    // 接口端口 (Verilog-2001 兼容)
    input  wire                                         clk,
    input  wire                                         rst_n,
    input  wire                                         i_valid,
    output wire                                         o_valid,
    input  wire signed [PARALLEL_FACTOR*DATA_WIDTH-1:0] parallel_inputs,
    input  wire signed [PARALLEL_FACTOR*DATA_WIDTH-1:0] parallel_weights,
    output wire signed [ACC_WIDTH-1:0]                  sum_of_products
);

    // =================================================================
    // == 1. 变量和寄存器声明
    // =================================================================
    
    // --- 2001修正: 循环变量必须在模块顶部声明为 integer ---
    integer k; 

    // --- 流水线数据寄存器 ---
    (* use_dsp = "yes" *) 
    reg signed [DATA_WIDTH*2+1:0] products_reg [0:PARALLEL_FACTOR-1];
    reg signed [ACC_WIDTH-1:0] sum_stage1_reg [0:PARALLEL_FACTOR/2-1];
    reg signed [ACC_WIDTH-1:0] sum_of_products_reg;
    
    // --- 控制流水线寄存器 ---
    reg [2:0] valid_pipeline_reg;

    // --- 组合逻辑线网 ---
    // --- 2001修正: generate块内使用的wire必须在外部声明为数组 ---
    wire signed [DATA_WIDTH+1:0] sub_result [0:PARALLEL_FACTOR-1];
    wire signed [DATA_WIDTH*2+1:0] products_comb [0:PARALLEL_FACTOR-1];
    wire signed [ACC_WIDTH-1:0] sum_stage1_comb [0:PARALLEL_FACTOR/2-1];
    wire signed [ACC_WIDTH-1:0] sum_stage2_comb [0:PARALLEL_FACTOR/4-1];
    wire signed [ACC_WIDTH-1:0] final_sum_comb;

    // --- 输出端口连接 ---
    assign sum_of_products = sum_of_products_reg;
    assign o_valid = valid_pipeline_reg[2];


    // =================================================================
    // == 2. 组合逻辑定义
    // =================================================================
    genvar i, s1, s2; // generate循环变量
    generate
        // --- 级1 的计算逻辑 (输入 -> 乘法) ---
        for (i = 0; i < PARALLEL_FACTOR; i = i + 1) begin : parallel_mac_units
            assign sub_result[i] = $signed(parallel_inputs[i*DATA_WIDTH +: DATA_WIDTH]) - $signed(INPUT_ZERO_POINT);
            assign products_comb[i] = sub_result[i] * $signed(parallel_weights[i*DATA_WIDTH +: DATA_WIDTH]);
        end

        // --- 级2 的计算逻辑 (第一级加法: 8 -> 4) ---
        for (s1 = 0; s1 < PARALLEL_FACTOR/2; s1 = s1 + 1) begin
            assign sum_stage1_comb[s1] = $signed(products_reg[2*s1]) + $signed(products_reg[2*s1+1]);
        end

        // --- 级3 的计算逻辑 (第二、三级加法: 4 -> 2 -> 1) ---
        for (s2 = 0; s2 < PARALLEL_FACTOR/4; s2 = s2 + 1) begin
            assign sum_stage2_comb[s2] = $signed(sum_stage1_reg[2*s2]) + $signed(sum_stage1_reg[2*s2+1]);
        end
    endgenerate
    assign final_sum_comb = $signed(sum_stage2_comb[0]) + $signed(sum_stage2_comb[1]);


    // =================================================================
    // == 3. 时序逻辑 (驱动所有流水线寄存器)
    // =================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // 复位所有寄存器
            // --- 2001修正: 使用预先声明的 integer 'k' ---
            for (k = 0; k < PARALLEL_FACTOR; k = k + 1) begin
                products_reg[k] <= 0;
            end
            for (k = 0; k < PARALLEL_FACTOR/2; k = k + 1) begin
                sum_stage1_reg[k] <= 0;
            end
            sum_of_products_reg <= 0;
            valid_pipeline_reg  <= 3'b0;
        end else begin
            // --- 数据流水线 ---
            // 级1 -> 级2
            // --- 2001修正: 使用预先声明的 integer 'k' ---
            for (k = 0; k < PARALLEL_FACTOR; k = k + 1) begin
                products_reg[k] <= products_comb[k];
            end

            // 级2 -> 级3
            // --- 2001修正: 使用预先声明的 integer 'k' ---
            for (k = 0; k < PARALLEL_FACTOR/2; k = k + 1) begin
                sum_stage1_reg[k] <= sum_stage1_comb[k];
            end
            
            // 级3 -> 输出
            sum_of_products_reg <= final_sum_comb;
            
            // --- 控制流水线 ---
            valid_pipeline_reg <= {valid_pipeline_reg[1:0], i_valid};
        end
    end

endmodule