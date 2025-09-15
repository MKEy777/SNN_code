`timescale 1ns / 1ps

module rom_sync #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 10,
    parameter MEM_FILE   = ""
)(
    input  wire                  clk,
    input  wire [ADDR_WIDTH-1:0] addr,
    output reg signed [DATA_WIDTH-1:0] data_out
);

    localparam MEM_DEPTH = 1 << ADDR_WIDTH;
    
    (* ram_style = "block" *) 
    reg [DATA_WIDTH-1:0] mem [0:MEM_DEPTH-1];

    initial begin
        if (MEM_FILE != "") begin
            $readmemh(MEM_FILE, mem);
        end
    end

    always @(posedge clk) begin
            data_out <= mem[addr];
    end

endmodule