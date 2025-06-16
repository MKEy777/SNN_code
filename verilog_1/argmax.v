`timescale 1ns / 1ps

module argmax #(
    parameter DATA_WIDTH = 32
)(
    // 系统信号
    input  wire                      clk,                // 时钟信号
    input  wire                      rst_n,              // 异步低电平复位信号
    
    // 握手信号
    input  wire                      i_valid,            // 输入有效信号
    output reg                       i_ready,            // 输入就绪信号
    input  wire                      o_ready,            // 输出就绪信号
    output reg                       o_valid,            // 输出有效信号
    
    // 数据端口
    input  wire signed [3*DATA_WIDTH-1:0] i_logits,      // 输入 logits 数据
    output reg [1:0]                 o_predicted_class   // 输出预测类别
);

    // 内部信号
    reg [1:0] predicted_class_reg;    // 寄存预测类别
    reg       data_processed;         // 数据处理标志

    // 组合逻辑：计算 argmax
    wire signed [DATA_WIDTH-1:0] logit_0 = i_logits[1*DATA_WIDTH-1 : 0*DATA_WIDTH];
    wire signed [DATA_WIDTH-1:0] logit_1 = i_logits[2*DATA_WIDTH-1 : 1*DATA_WIDTH];
    wire signed [DATA_WIDTH-1:0] logit_2 = i_logits[3*DATA_WIDTH-1 : 2*DATA_WIDTH];
    
    wire [1:0] predicted_class_comb = (logit_0 >= logit_1 && logit_0 >= logit_2) ? 2'd0 :
                                      (logit_1 >  logit_0 && logit_1 >= logit_2) ? 2'd1 :
                                                                            2'd2;

    // 时序逻辑：寄存器更新与握手信号管理
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_valid <= 1'b0;
            i_ready <= 1'b1;          // 初始状态为准备好接收数据
            predicted_class_reg <= 2'd0;
            data_processed <= 1'b0;
        end else begin
            if (i_valid && i_ready) begin
                // 接收输入数据，计算并存储结果
                predicted_class_reg <= predicted_class_comb;
                data_processed <= 1'b1;
                i_ready <= 1'b0;      // 表示模块忙碌
            end else if (data_processed && o_ready) begin
                // 下游准备好，输出结果
                o_valid <= 1'b1;
                o_predicted_class <= predicted_class_reg;
                data_processed <= 1'b0;
            end else if (o_valid && o_ready) begin
                // 输出被下游接收，复位状态
                o_valid <= 1'b0;
                i_ready <= 1'b1;      // 准备接收下一个数据
            end
        end
    end

endmodule