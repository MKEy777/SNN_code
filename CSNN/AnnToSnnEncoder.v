`timescale 1ns / 1ps

module AnnToSnnEncoder_serial #(
    // --- 结构参数 ---
    parameter VEC_LEN       = 320,      
    parameter PIXEL_VEC_LEN = 16,       
    parameter NUM_PIXELS    = 20,      
    parameter DATA_W        = 8,        
    parameter TIME_W        = 32,      

    // --- 编码常量 (来自训练) ---
    parameter signed [TIME_W-1:0] T_MAX             = 32'h7FFFFFFF,
    parameter signed [TIME_W-1:0] TIME_OFFSET       = 32'h0000_0001,
    parameter signed [15:0]       TIME_SCALE_MULT = 16'h0160,
    parameter SHIFT_BITS      = 15
) (
    // --- 系统接口 ---
    input  wire                      clk,
    input  wire                      rst_n,

    // --- 流式输入接口  ---
    input  wire                      i_pixel_valid,
    input  wire signed [PIXEL_VEC_LEN*DATA_W-1:0] i_pixel_vec, 

    // --- 串行脉冲输出接口  ---
    output reg                       o_spike_valid,
    input  wire                      i_spike_ack,
    output reg signed [TIME_W-1:0]   o_spike_time,
    output reg [$clog2(VEC_LEN)-1:0] o_spike_addr,
    
    // --- 状态输出 ---
    output wire                      o_busy,
    output wire                      o_last_pixel_sent 
);

    // --- FSM 状态定义 ---
    localparam S_IDLE        = 3'b000; 
    localparam S_CALC_LOAD   = 3'b001; 
    localparam S_SEND_SPIKE  = 3'b010; 
    localparam S_FINISHING   = 3'b011; 

    reg [2:0] state, next_state;

    // --- 内部寄存器 ---
    reg [$clog2(NUM_PIXELS)-1:0] pixel_receive_cnt; 
    reg [$clog2(PIXEL_VEC_LEN)-1:0] channel_send_cnt;   
    reg signed [TIME_W-1:0] current_pixel_spikes [0:PIXEL_VEC_LEN-1];
    wire signed [TIME_W-1:0] calculated_spikes [0:PIXEL_VEC_LEN-1];
    genvar i;
    generate
        for (i = 0; i < PIXEL_VEC_LEN; i = i + 1) begin : PIXEL_ENCODE_LOGIC
            wire signed [DATA_W-1:0] x_q;
            wire signed [23:0]       product; 
            wire signed [TIME_W-1:0] scaled_val;

            assign x_q        = i_pixel_vec[(i+1)*DATA_W-1 -: DATA_W];
            assign product    = TIME_SCALE_MULT * x_q;
            assign scaled_val = product >>> SHIFT_BITS;
            assign calculated_spikes[i] = TIME_OFFSET - scaled_val;
        end
    endgenerate

    // --- 状态机 - 组合逻辑部分 (无变化) ---
    always @(*) begin
        next_state    = state;
        o_spike_valid = 1'b0;
        o_spike_time  = {TIME_W{1'bx}};
        o_spike_addr  = {$clog2(VEC_LEN){1'bx}};
        
        case (state)
            S_IDLE: begin
                if (i_pixel_valid) begin
                    next_state = S_CALC_LOAD;
                end
            end
            
            S_CALC_LOAD: begin
                next_state = S_SEND_SPIKE;
            end

            S_SEND_SPIKE: begin
                if (current_pixel_spikes[channel_send_cnt] < T_MAX) begin
                    o_spike_valid = 1'b1;
                    o_spike_time  = current_pixel_spikes[channel_send_cnt];
                    o_spike_addr  = pixel_receive_cnt * PIXEL_VEC_LEN + channel_send_cnt;
                    
                    if (i_spike_ack) begin
                        if (channel_send_cnt == PIXEL_VEC_LEN - 1) begin
                            if (pixel_receive_cnt == NUM_PIXELS - 1) begin
                                next_state = S_FINISHING;
                            end else begin
                                next_state = S_IDLE;
                            end
                        end else begin
                            next_state = S_SEND_SPIKE;
                        end
                    end else begin
                        next_state = S_SEND_SPIKE;
                    end
                end else begin // 当前脉冲无效
                    if (channel_send_cnt == PIXEL_VEC_LEN - 1) begin
                        if (pixel_receive_cnt == NUM_PIXELS - 1) begin
                            next_state = S_FINISHING;
                        end else begin
                            next_state = S_IDLE;
                        end
                    end else begin
                        next_state = S_SEND_SPIKE;
                    end
                end
            end

            S_FINISHING: begin
                next_state = S_IDLE;
            end
            
            default: begin
                next_state = S_IDLE;
            end
        endcase
    end
    integer ch;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            pixel_receive_cnt <= 0;
            channel_send_cnt <= 0;
            // 在复位时也必须初始化寄存器阵列
            for (ch = 0; ch < PIXEL_VEC_LEN; ch = ch + 1) begin
                current_pixel_spikes[ch] <= 0;
            end
        end else begin
            state <= next_state; // 状态转移总是发生
            case (state)
                S_IDLE: begin
                    if (i_pixel_valid) begin
                        if (pixel_receive_cnt == NUM_PIXELS - 1) begin
                            pixel_receive_cnt <= 0;
                        end else begin
                            pixel_receive_cnt <= pixel_receive_cnt + 1;
                        end
                    end
                end

                S_CALC_LOAD: begin
                    for (ch = 0; ch < PIXEL_VEC_LEN; ch = ch + 1) begin
                        current_pixel_spikes[ch] <= calculated_spikes[ch];
                    end
                    channel_send_cnt <= 0; 
                end
                
                S_SEND_SPIKE: begin
                    if (current_pixel_spikes[channel_send_cnt] >= T_MAX || (o_spike_valid && i_spike_ack)) begin
                        if (channel_send_cnt != PIXEL_VEC_LEN - 1) begin
                            channel_send_cnt <= channel_send_cnt + 1;
                        end
                    end
                end
                
                S_FINISHING: begin
                    pixel_receive_cnt <= 0;
                    channel_send_cnt  <= 0;
                end
            endcase
        end
    end
    assign o_busy = (state != S_IDLE);
    assign o_last_pixel_sent = o_spike_valid && (channel_send_cnt == PIXEL_VEC_LEN - 1) && (pixel_receive_cnt == NUM_PIXELS - 1);

endmodule