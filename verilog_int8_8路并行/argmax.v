`timescale 1ns / 1ps
//
// 功能: 找出三个输入logits中值最大的一个的索引。
//       DATA_WIDTH 参数使其可以灵活支持不同位宽的输入。
//
module argmax #(
    parameter DATA_WIDTH = 16 
)(
    // --- 系统信号 ---
    input  wire                      clk,
    input  wire                      rst_n,
    
    // --- 握手信号 ---
    input  wire                      i_valid,
    output wire                      i_ready,
    input  wire                      o_ready,
    output wire                      o_valid,
    
    // --- 数据端口 ---
    input  wire signed [3*DATA_WIDTH-1:0] i_logits,
    output wire [1:0]                     o_predicted_class
);

    // --- 组合逻辑数据路径 ---
    // 根据传入的DATA_WIDTH自动切分输入总线
    wire signed [DATA_WIDTH-1:0] logit_0 = i_logits[1*DATA_WIDTH-1 : 0*DATA_WIDTH];
    wire signed [DATA_WIDTH-1:0] logit_1 = i_logits[2*DATA_WIDTH-1 : 1*DATA_WIDTH];
    wire signed [DATA_WIDTH-1:0] logit_2 = i_logits[3*DATA_WIDTH-1 : 2*DATA_WIDTH];

    assign o_predicted_class = (logit_0 >= logit_1 && logit_0 >= logit_2) ? 2'd0 :
                               (logit_1 >  logit_0 && logit_1 >= logit_2) ? 2'd1 :
                                                                            2'd2;
                                                                            
    // --- 组合逻辑握手路径 ---
    assign o_valid = i_valid;
    assign i_ready = o_ready;

endmodule