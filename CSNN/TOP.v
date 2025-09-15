`timescale 1ns / 1ps

module top (
    input  wire                      clk_cnn,        
    input  wire                      clk_snn,         
    input  wire                      rst_n,           
    input  wire                      start_inference,
    output wire                      inference_done,
    output wire [1:0]                predicted_class,
    input  wire                      bram_we,
    input  wire [8:0]                bram_addr,
    input  wire signed [7:0]         bram_wdata
);

    wire cnn_to_snn_spike_valid;
    wire cnn_to_snn_spike_ack;
    wire signed [31:0] cnn_to_snn_spike_time;
    wire [8:0]         cnn_to_snn_spike_addr;
    wire cnn_done;

    // --- 同步后的控制信号 ---
    wire start_inference_synced_to_cnn; 
    wire cnn_to_snn_spike_valid_synced_to_snn;
    wire cnn_to_snn_spike_ack_synced_to_cnn;  
    wire cnn_done_synced_to_snn;              


    two_flop_synchronizer start_sync_inst (
        .clk_dest(clk_cnn),
        .rst_n(rst_n),
        .d_in(start_inference),
        .d_out(start_inference_synced_to_cnn)
    );

    two_flop_synchronizer valid_sync_inst (
        .clk_dest(clk_snn),
        .rst_n(rst_n),
        .d_in(cnn_to_snn_spike_valid),
        .d_out(cnn_to_snn_spike_valid_synced_to_snn)
    );

    two_flop_synchronizer ack_sync_inst (
        .clk_dest(clk_cnn),
        .rst_n(rst_n),
        .d_in(cnn_to_snn_spike_ack),
        .d_out(cnn_to_snn_spike_ack_synced_to_cnn)
    );

    two_flop_synchronizer cnn_done_sync_inst (
        .clk_dest(clk_snn),
        .rst_n(rst_n),
        .d_in(cnn_done),
        .d_out(cnn_done_synced_to_snn)
    );

    cnn_top_streaming cnn_instance (
        .clk(clk_cnn), 
        .rst_n(rst_n), 
        .i_start_inference(start_inference_synced_to_cnn), 
        .i_bram_we(bram_we), 
        .i_bram_addr(bram_addr), 
        .i_bram_wdata(bram_wdata),
        
        // --- 连接到串行脉冲输出接口 ---
        .o_spike_valid(cnn_to_snn_spike_valid),
        .i_spike_ack(cnn_to_snn_spike_ack_synced_to_cnn),
        .o_spike_time(cnn_to_snn_spike_time),
        .o_spike_addr(cnn_to_snn_spike_addr),
        .o_cnn_done(cnn_done)
    );
    
    snn_top snn_instance (
        .clk(clk_snn), 
        .rst_n(rst_n),
        .i_cnn_spike_valid(cnn_to_snn_spike_valid_synced_to_snn), 
        .o_cnn_spike_ack(cnn_to_snn_spike_ack),
        .i_cnn_spike_time(cnn_to_snn_spike_time),   
        .i_cnn_spike_addr(cnn_to_snn_spike_addr),  
        .i_cnn_last_pixel_sent(cnn_done_synced_to_snn), 
        .o_predicted_class(predicted_class), 
        .o_inference_done(inference_done)
    );

endmodule
`timescale 1ns / 1ps

module two_flop_synchronizer (
    input  wire clk_dest, 
    input  wire rst_n,    
    input  wire d_in,     
    output wire d_out     
);

    (* async_reg = "true" *) reg d_meta;
    (* async_reg = "true" *) reg d_sync;
    
    always @(posedge clk_dest or negedge rst_n) begin
        if (!rst_n) begin
            d_meta <= 1'b0;
            d_sync <= 1'b0;
        end else begin
            d_meta <= d_in;
            d_sync <= d_meta;
        end
    end

    assign d_out = d_sync;

endmodule