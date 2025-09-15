`timescale 1ns / 1ps

module Stream_To_Parallel #(
    parameter VEC_LEN = 3,
    parameter DATA_W  = 32
) (
    input wire clk,
    input wire rst_n,

    // Stream Input
    input wire                      i_valid,
    input wire signed [DATA_W-1:0]  i_data,
    input wire                      i_last,
    output wire                     o_ack,

    // Parallel Output
    output reg signed [VEC_LEN*DATA_W-1:0] o_data_flat,
    output reg                             o_valid_out
);
    reg [$clog2(VEC_LEN)-1:0] write_ptr;
    reg signed [DATA_W-1:0]   data_reg [0:VEC_LEN-1];
    reg busy;
    
    assign o_ack = i_valid && !busy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_ptr <= 0;
            o_valid_out <= 1'b0;
            busy <= 1'b0;
        end else begin
            o_valid_out <= 1'b0; // Default to pulse
            if (o_ack) begin
                data_reg[write_ptr] <= i_data;
                if (i_last) begin
                    o_valid_out <= 1'b1; // Pulse valid high for one cycle
                    write_ptr <= 0;
                    busy <= 1'b0;
                end else begin
                    write_ptr <= write_ptr + 1;
                    busy <= 1'b1;
                end
            end else if (busy && write_ptr == 0) begin // Acknowledge has been seen by master
                 busy <= 1'b0;
            end
        end
    end
    integer i;
    always @(*) begin
        for (i = 0; i < VEC_LEN; i = i + 1) begin
            o_data_flat[(i+1)*DATA_W-1 -: DATA_W] = data_reg[i];
        end
    end
endmodule