`timescale 1ns / 1ps

module line_buffer_3x3 #(
    parameter DATA_W = 8,
    parameter IMG_W  = 9,
    parameter K_DIM  = 3   
) (
    input  wire                      clk,
    input  wire                      rst_n,   
    input  wire                      i_en,    
    input  wire signed [DATA_W-1:0]  i_data,
    output reg signed [(K_DIM*K_DIM*DATA_W)-1:0] o_win_flat,
    output reg                       o_valid
);
    localparam K_SZ = K_DIM * K_DIM;

    // 内部寄存器
    reg signed [DATA_W-1:0] line_buf1 [0:IMG_W-1];
    reg signed [DATA_W-1:0] line_buf2 [0:IMG_W-1];
    reg signed [DATA_W-1:0] win_regs [0:K_DIM-1][0:K_DIM-1];
    reg [$clog2(IMG_W):0] write_ptr;  
    reg [$clog2(IMG_W*IMG_W):0] row_cnt;
    integer r, c;

    // 内部连线，用于组合逻辑计算
    wire signed [(K_DIM*K_DIM*DATA_W)-1:0] win_flat_internal;
    wire valid_internal;

    // 核心逻辑: 更新内部状态
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_ptr <= 0;
            row_cnt   <= 0;
            for (r = 0; r < IMG_W; r = r + 1) begin
                line_buf1[r] <= 9;
                line_buf2[r] <= 9;
            end
            for (r = 0; r < K_DIM; r = r + 1) begin
                for (c = 0; c < K_DIM; c = c + 1) begin
                    win_regs[r][c] <= 9;
                end
            end
        end else if (i_en) begin
            // 窗口移位
            for (r = 0; r < K_DIM; r = r + 1) begin
                win_regs[r][0] <= win_regs[r][1];
                win_regs[r][1] <= win_regs[r][2];
            end
            // 加载新列
            win_regs[0][2] <= line_buf2[write_ptr];
            win_regs[1][2] <= line_buf1[write_ptr];
            win_regs[2][2] <= i_data;
            // 更新行缓冲
            line_buf2[write_ptr] <= line_buf1[write_ptr];
            line_buf1[write_ptr] <= i_data;
            // 计数器更新
            if (write_ptr == IMG_W - 1) begin
                write_ptr <= 0;
                row_cnt   <= row_cnt + 1;
            end else begin
                write_ptr <= write_ptr + 1;
            end
        end
    end

    assign valid_internal = (row_cnt >= (K_DIM-1)) && (write_ptr >= (K_DIM-1));
    
    genvar gw;
    generate
        for (gw = 0; gw < K_SZ; gw = gw + 1) begin : pack_output_window
            assign win_flat_internal[(gw*DATA_W) +: DATA_W] = win_regs[gw / K_DIM][gw % K_DIM];
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_valid    <= 1'b0;
            o_win_flat <= 0;
        end else if (i_en) begin
            o_valid    <= valid_internal;
            o_win_flat <= win_flat_internal;
        end else 
            o_valid    <= 1'b0;
    end

endmodule