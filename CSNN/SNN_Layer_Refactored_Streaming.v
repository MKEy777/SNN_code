`timescale 1ns / 1ps

module SNN_Layer_Refactored_Streaming #(
    parameter IS_OUTPUT_LAYER   = 0,
    parameter IN_NEURONS        = 64,
    parameter OUT_NEURONS       = 32,
    parameter INPUT_FIFO_DEPTH  = 16,
    parameter OUTPUT_FIFO_DEPTH = 16,
    parameter TIME_W            = 32,
    parameter WEIGHT_W          = 8,
    parameter ACC_W             = 32
) (
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire                      i_clk_enable,
    output wire                      o_done,           
    input  wire                      i_spike_valid,
    input  wire                      i_last_spike,     
    input  wire signed [TIME_W-1:0]  i_spike_time,
    input  wire [$clog2(IN_NEURONS)-1:0] i_spike_addr, 
    output wire                      o_spike_ack,     
    output wire                      o_result_valid,
    output wire signed [ACC_W-1:0]   o_result_data,
    output wire                      o_last_result,
    input  wire                      i_result_ack,
    output wire [$clog2(IN_NEURONS*OUT_NEURONS)-1:0]  weight_ram_addr,
    input  wire signed [WEIGHT_W-1:0]                 weight_ram_rdata,
    output wire [$clog2(OUT_NEURONS)-1:0]             param_ram_addr,
    input  wire signed [ACC_W-1:0]                    param_ram_rdata
);

    localparam SPIKE_DATA_IN_W = 1 + TIME_W + $clog2(IN_NEURONS); // 增加1位用于i_last_spike
    wire in_fifo_full;
    wire in_fifo_empty;
    wire [SPIKE_DATA_IN_W-1:0] in_fifo_rdata;

    // 输出FIFO相关
    localparam SPIKE_DATA_OUT_W = 1 + ACC_W; // 增加1位用于o_last_result
    wire out_fifo_full;
    wire out_fifo_empty;
    wire [SPIKE_DATA_OUT_W-1:0] out_fifo_rdata;

    // 核心模块相关
    wire core_spike_ack;
    wire core_result_valid;
    wire signed [ACC_W-1:0] core_result_data;
    wire core_last_result;
    wire core_layer_done;

    // --- 模块实例化 ---

    // 1. 输入脉冲FIFO
    synchronous_fifo #(
        .DATA_WIDTH(SPIKE_DATA_IN_W),
        .DEPTH(INPUT_FIFO_DEPTH)
    ) Input_FIFO_Inst (
        .clk(clk), .rst_n(rst_n),
        .i_wr_en(i_spike_valid && o_spike_ack),
        .i_wdata({i_last_spike, i_spike_time, i_spike_addr}),
        .o_full(in_fifo_full),
        .i_rd_en(core_spike_ack && !in_fifo_empty),
        .o_rdata(in_fifo_rdata),
        .o_empty(in_fifo_empty)
    );

    // 2. SNN计算核心 (正确的实例化)
    SNN_Core_Streaming #(
        .IN_NEURONS(IN_NEURONS),
        .OUT_NEURONS(OUT_NEURONS),
        .TIME_W(TIME_W),
        .WEIGHT_W(WEIGHT_W),
        .ACC_W(ACC_W),
        .IS_OUTPUT_LAYER(IS_OUTPUT_LAYER)
    ) Core_Inst (
        .clk(clk), .rst_n(rst_n), .i_clk_enable(i_clk_enable),
        .i_spike_valid(!in_fifo_empty),
        .i_spike_time(in_fifo_rdata[TIME_W + $clog2(IN_NEURONS) - 1 -: TIME_W]),
        .i_spike_addr(in_fifo_rdata[$clog2(IN_NEURONS)-1:0]),
        .i_last_spike(in_fifo_rdata[SPIKE_DATA_IN_W-1]),
        .o_spike_ack(core_spike_ack),
        .o_result_valid(core_result_valid),
        .o_result_data(core_result_data),
        .o_last_result(core_last_result),
        .i_result_ack(!out_fifo_full),
        .weight_ram_addr(weight_ram_addr),
        .weight_ram_rdata(weight_ram_rdata),
        .param_ram_addr(param_ram_addr),
        .param_ram_rdata(param_ram_rdata),
        .o_layer_done(core_layer_done)
    );

    // 3. 输出结果FIFO
    synchronous_fifo #(
        .DATA_WIDTH(SPIKE_DATA_OUT_W),
        .DEPTH(OUTPUT_FIFO_DEPTH)
    ) Output_FIFO_Inst (
        .clk(clk), .rst_n(rst_n),
        .i_wr_en(core_result_valid && !out_fifo_full),
        .i_wdata({core_last_result, core_result_data}),
        .o_full(out_fifo_full),
        .i_rd_en(o_result_valid && i_result_ack),
        .o_rdata(out_fifo_rdata),
        .o_empty(out_fifo_empty)
    );

    // --- 输出逻辑 ---
    // 向上游反馈：只要我的输入FIFO没满，就可以接收新脉冲
    assign o_spike_ack = !in_fifo_full;

    // 向下游发送：只要我的输出FIFO不空，就有有效数据
    assign o_result_valid = !out_fifo_empty;
    assign {o_last_result, o_result_data} = out_fifo_rdata;
    
    // --- 完成信号逻辑 ---
    reg core_done_latch;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            core_done_latch <= 1'b0;
        end else if (core_layer_done) begin
            core_done_latch <= 1'b1; // 锁存核心完成信号
        end else if (o_done) begin
            core_done_latch <= 1'b0; // 在任务完成后复位，为下一次任务准备
        end
    end

    // 当核心完成计算 且 输出FIFO已空时，整个层的任务才算完成
    assign o_done = core_done_latch && out_fifo_empty;

endmodule