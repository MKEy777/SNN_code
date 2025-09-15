`timescale 1ns / 1ps

module depthwise_conv_unit #(
    parameter DATA_W = 8,
    parameter K_DIM  = 3,
    parameter ACC_W  = 32
) (
    input clk,
    input rst_n,
    input i_valid,

    input signed [K_DIM*K_DIM*DATA_W-1:0] i_win_flat,
    input signed [K_DIM*K_DIM*DATA_W-1:0] i_kernel_flat,
    
    output reg signed [ACC_W-1:0] o_acc,
    output reg o_valid
);
    localparam K_SZ = K_DIM * K_DIM;
    
    // -- Stage 0: Combinational Unpack --
    wire signed [DATA_W-1:0] i_win [0:K_SZ-1];
    wire signed [DATA_W-1:0] i_kernel [0:K_SZ-1];
    genvar j;
    generate
        for (j = 0; j < K_SZ; j = j + 1) begin : unpack_ports
            assign i_win[j]    = i_win_flat   [(j+1)*DATA_W-1 : j*DATA_W];
            assign i_kernel[j] = i_kernel_flat[(j+1)*DATA_W-1 : j*DATA_W];
        end
    endgenerate

    // -- Stage 1: Combinational Multiply & Register --
    wire signed [DATA_W*2-1:0] products [0:K_SZ-1];
    reg  signed [DATA_W*2-1:0] products_reg [0:K_SZ-1];
    reg                        s1_valid; 
    genvar i;
    integer k;
    generate
        for(i = 0; i < K_SZ; i = i + 1) begin
            assign products[i] = $signed(i_win[i]) * $signed(i_kernel[i]);
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
            for (k = 0; k < K_SZ; k = k + 1) begin
                products_reg[k] <= 0;
            end
        end else begin
            s1_valid <= i_valid;
            if (i_valid) begin
                for (k = 0; k < K_SZ; k = k + 1) begin
                    products_reg[k] <= products[k];
                end
            end
        end
    end
    
    reg signed [ACC_W-1:0] partial_sum1, partial_sum2, partial_sum3;
    reg s2_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            partial_sum1 <= 0;
            partial_sum2 <= 0;
            partial_sum3 <= 0;
            s2_valid     <= 1'b0;
        end else begin
            s2_valid <= s1_valid; 
            if (s1_valid) begin
                partial_sum1 <= products_reg[0] + products_reg[1] + products_reg[2];
                partial_sum2 <= products_reg[3] + products_reg[4] + products_reg[5];
                partial_sum3 <= products_reg[6] + products_reg[7] + products_reg[8];
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_acc   <= {ACC_W{1'b0}};
            o_valid <= 1'b0;
        end else begin
            o_valid <= s2_valid; 
            if (s2_valid) begin
                o_acc <= partial_sum1 + partial_sum2 + partial_sum3;
            end else begin
                o_acc <= {ACC_W{1'b0}};
            end
        end
    end
    
endmodule