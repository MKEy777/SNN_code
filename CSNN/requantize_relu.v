`timescale 1ns / 1ps

module requantize_relu #(
    parameter IN_W       = 32,
    parameter BIAS_W     = 32,
    parameter OUT_W      = 8,
    parameter SCALE_W    = 32,
    parameter SHIFT_BITS = 31
) (
    input clk,
    input rst_n,
    input i_valid,
    input signed [IN_W-1:0]   i_acc,
    input signed [BIAS_W-1:0] i_bias,
    input signed [SCALE_W-1:0] i_scale,
    input signed [OUT_W-1:0]  i_zero_point,
    
    output reg signed [OUT_W-1:0] o_data,
    output reg o_valid
);
    localparam Q_MIN = -(2**(OUT_W-1));
    localparam Q_MAX = (2**(OUT_W-1)) - 1;

    // --- 流水线寄存器定义 ---

    // Pipeline Stage 1: 输入锁存
    reg p1_valid;
    reg signed [IN_W-1:0]   p1_acc;
    reg signed [BIAS_W-1:0] p1_bias;
    reg signed [SCALE_W-1:0] p1_scale;
    reg signed [OUT_W-1:0]  p1_zero_point;

    // Pipeline Stage 2: 偏置加法结果
    reg p2_valid;
    reg signed [IN_W:0]      p2_acc_biased;
    reg signed [SCALE_W-1:0] p2_scale;
    reg signed [OUT_W-1:0]   p2_zero_point;

    // Pipeline Stage 3: 乘法结果
    reg p3_valid;
    reg signed [IN_W+SCALE_W:0] p3_scaled_acc;
    reg signed [OUT_W-1:0]      p3_zero_point;
    wire signed [IN_W:0] shifted_acc;
    wire signed [IN_W:0] final_acc;

    // 使用assign语句定义组合逻辑
    assign shifted_acc = p3_scaled_acc >>> SHIFT_BITS;
    assign final_acc   = shifted_acc + $signed(p3_zero_point);


    // --- 流水线逻辑 ---

    // Stage 1: 输入锁存 always 块
    always @(posedge clk) begin
        if (!rst_n) begin
            p1_valid <= 1'b0;
        end else begin
            p1_valid <= i_valid;
            if (i_valid) begin
                p1_acc        <= i_acc;
                p1_bias       <= i_bias;
                p1_scale      <= i_scale;
                p1_zero_point <= i_zero_point;
            end
        end
    end

    // Stage 2: 偏置加法 always 块
    always @(posedge clk) begin
        if (!rst_n) begin
            p2_valid <= 1'b0;
        end else begin
            p2_valid <= p1_valid;
            if (p1_valid) begin
                p2_acc_biased <= $signed(p1_acc) + $signed(p1_bias);
                p2_scale      <= p1_scale;
                p2_zero_point <= p1_zero_point;
            end
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            p3_valid <= 1'b0;
        end else begin
            p3_valid <= p2_valid;
            if (p2_valid) begin
                p3_scaled_acc <= p2_acc_biased * $signed(p2_scale);
                p3_zero_point <= p2_zero_point;
            end
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            o_valid <= 1'b0;
            o_data  <= {OUT_W{1'b0}};
        end else begin
            o_valid <= p3_valid;
            if (p3_valid) begin
                if (final_acc < 0) begin
                    o_data <= {OUT_W{1'b0}}; 
                end else if (final_acc > Q_MAX) begin
                    o_data <= Q_MAX; 
                end else begin
                    o_data <= final_acc[OUT_W-1:0];
                end
            end
        end
    end

endmodule