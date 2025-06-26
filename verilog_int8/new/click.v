`timescale 1ns / 1ps

/**
 * @brief 标准 Click 单元，带有一个用于监控的 fire 输出信号
 * @details
 * - 核心逻辑使用电平敏感锁存器，安全、可综合且无毛刺风险。
 * - `fire` 信号是一个"观察窗"，它显示了状态即将改变的条件，但它本身
 * 并不驱动状态更新，从而保证了设计的健壮性。
 */
module click_standard_with_fire_monitor (
    // 输入端口
    input  wire in_req,    // 从上游来的请求信号
    input  wire out_ack,   // 从下游来的应答信号
    input  wire rst_n,     // 异步复位，低电平有效

    // 输出端口
    output wire out_req,   // 向下游发的请求信号
    output wire in_ack,    // 向上游发的应答信号
    output wire fire       // 用于监控的触发信号
);

    // 内部状态寄存器
    reg s;

    always @(*) begin
        if (!rst_n) begin
            s = 1'b0;
        end else begin
            if (s == out_ack) begin
                s = in_req;
            end else begin
                s = s; 
            end
        end
    end

    assign out_req = s;
    assign in_ack  = s;
    assign fire = (s == out_ack) & (s != in_req);

endmodule