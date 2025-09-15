`timescale 1ns / 1ps

module synchronous_fifo #(
    parameter DATA_WIDTH = 32,
    parameter DEPTH      = 16
) (
    input  wire                      clk,
    input  wire                      rst_n,

    // Write Interface
    input  wire                      i_wr_en,
    input  wire [DATA_WIDTH-1:0]     i_wdata,
    output wire                      o_full,

    // Read Interface
    input  wire                      i_rd_en,
    output wire [DATA_WIDTH-1:0]     o_rdata,
    output wire                      o_empty
);
    localparam ADDR_WIDTH = $clog2(DEPTH);

    // 存储器阵列
    (* ram_style = "block" *) // 指导综合器使用BRAM
    reg [DATA_WIDTH-1:0] mem[0:DEPTH-1];
    
    // 读写指针，比地址位宽多1位，用于区分满/空状态
    reg [ADDR_WIDTH:0]   wr_ptr;
    reg [ADDR_WIDTH:0]   rd_ptr;

    integer i;
    initial begin
        for (i = 0; i < DEPTH; i = i + 1) begin
            mem[i] = 0;
        end
    end

    // --- 核心逻辑 ---
    // 指针预计算
    wire [ADDR_WIDTH:0] wr_ptr_next = wr_ptr + 1;
    wire [ADDR_WIDTH:0] rd_ptr_next = rd_ptr + 1;

    // 满/空状态判断 (逻辑保持不变)
    assign o_full  = (wr_ptr_next == {~rd_ptr[ADDR_WIDTH], rd_ptr[ADDR_WIDTH-1:0]});
    assign o_empty = (wr_ptr == rd_ptr);
    
    // 读数据端口 (行为保持不变)
    assign o_rdata = mem[rd_ptr[ADDR_WIDTH-1:0]];

    // --- 写逻辑 (修改为同步复位) ---
    always @(posedge clk) begin
        if (!rst_n) begin
            wr_ptr <= 0;
        end else if (i_wr_en && !o_full) begin
            mem[wr_ptr[ADDR_WIDTH-1:0]] <= i_wdata;
            wr_ptr <= wr_ptr_next;
        end
    end

    // --- 读逻辑 (修改为同步复位) ---
    always @(posedge clk) begin
        if (!rst_n) begin
            rd_ptr <= 0;
        end else if (i_rd_en && !o_empty) begin
            rd_ptr <= rd_ptr_next;
        end
    end
    
endmodule