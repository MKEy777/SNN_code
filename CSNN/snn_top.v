`timescale 1ns / 1ps

module snn_top (
    input  wire                      clk,
    input  wire                      rst_n,

    input  wire                      i_cnn_spike_valid,
    input  wire signed [31:0]        i_cnn_spike_time,
    input  wire [8:0]                i_cnn_spike_addr,
    input  wire                      i_cnn_last_pixel_sent,
    output wire                      o_cnn_spike_ack,

    output wire [1:0]                o_predicted_class,
    output wire                      o_inference_done
);
    localparam SNN_L1_IN = 320, SNN_L1_OUT = 64;
    localparam SNN_L2_IN = 64,  SNN_L2_OUT = 32;
    localparam SNN_L3_IN = 32,  SNN_L3_OUT = 3;
    localparam WEIGHT_W = 8, ACC_W = 32;

    // --- 全局时钟使能逻辑 ---
    reg inference_busy;
    wire argmax_done;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            inference_busy <= 1'b0;
        end else if (i_cnn_spike_valid && !inference_busy) begin
            // SET: 当不在忙且第一个脉冲到来时，开始置位"忙"信号
            inference_busy <= 1'b1;
        end else if (argmax_done) begin
            // RESET: 当ArgMax完成时，表示整个推理结束，复位"忙"信号
            inference_busy <= 1'b0;
        end
    end

    // --- Pipeline Stage Wires ---
    wire s1_out_valid, s1_out_last, s1_out_ack, snn1_done;
    wire signed [ACC_W-1:0] s1_out_data;
    
    wire b2_out_req, b2_out_ack, b2_out_req_type, bridge2_done;
    wire signed [ACC_W-1:0] b2_out_spike_time;
    wire [$clog2(SNN_L2_IN)-1:0] b2_out_spike_addr;

    wire s2_out_valid, s2_out_last, s2_out_ack, snn2_done;
    wire signed [ACC_W-1:0] s2_out_data;
    
    wire b3_out_req, b3_out_ack, b3_out_req_type, bridge3_done;
    wire signed [ACC_W-1:0] b3_out_spike_time;
    wire [$clog2(SNN_L3_IN)-1:0] b3_out_spike_addr;
    
    wire s3_out_valid, s3_out_last, s3_out_ack, snn3_done;
    wire signed [ACC_W-1:0] s3_out_data;

    wire s2p_valid_out;
    wire signed [SNN_L3_OUT*ACC_W-1:0] s2p_data_flat;
    

    // --- Memory Interface Wires ---
    wire [$clog2(SNN_L1_IN*SNN_L1_OUT)-1:0] w1_addr; wire signed [WEIGHT_W-1:0] w1_rdata;
    wire [$clog2(SNN_L1_OUT)-1:0] p1_addr;           wire signed [ACC_W-1:0]    p1_rdata;
    wire [$clog2(SNN_L2_IN*SNN_L2_OUT)-1:0] w2_addr; wire signed [WEIGHT_W-1:0] w2_rdata;
    wire [$clog2(SNN_L2_OUT)-1:0] p2_addr;           wire signed [ACC_W-1:0]    p2_rdata;
    wire [$clog2(SNN_L3_IN*SNN_L3_OUT)-1:0] w3_addr; wire signed [WEIGHT_W-1:0] w3_rdata;
    wire [$clog2(SNN_L3_OUT)-1:0] p3_addr;           wire signed [ACC_W-1:0]    p3_rdata;
    
    // --- Memory Instantiations (ROMs Only) ---
    rom_sync #(.DATA_WIDTH(WEIGHT_W), .ADDR_WIDTH($clog2(SNN_L1_IN*SNN_L1_OUT)), .MEM_FILE("snn_layer1_weights.mem"))
    Weight_ROM1 (.clk(clk),.addr(w1_addr), .data_out(w1_rdata));
    rom_sync #(.DATA_WIDTH(ACC_W), .ADDR_WIDTH($clog2(SNN_L1_OUT)), .MEM_FILE("snn_layer1_d_i.mem"))
    Param_ROM1  (.clk(clk),.addr(p1_addr), .data_out(p1_rdata));
    rom_sync #(.DATA_WIDTH(WEIGHT_W), .ADDR_WIDTH($clog2(SNN_L2_IN*SNN_L2_OUT)), .MEM_FILE("snn_layer2_weights.mem"))
    Weight_ROM2 (.clk(clk), .addr(w2_addr), .data_out(w2_rdata));
    rom_sync #(.DATA_WIDTH(ACC_W), .ADDR_WIDTH($clog2(SNN_L2_OUT)), .MEM_FILE("snn_layer2_d_i.mem"))
    Param_ROM2  (.clk(clk), .addr(p2_addr), .data_out(p2_rdata));
    rom_sync #(.DATA_WIDTH(WEIGHT_W), .ADDR_WIDTH($clog2(SNN_L3_IN*SNN_L3_OUT)), .MEM_FILE("snn_layer3_weights.mem"))
    Weight_ROM3 (.clk(clk), .addr(w3_addr), .data_out(w3_rdata));
    rom_sync #(.DATA_WIDTH(ACC_W), .ADDR_WIDTH($clog2(SNN_L3_OUT)), .MEM_FILE("snn_layer3_bias.mem"))
    Param_ROM3  (.clk(clk), .addr(p3_addr), .data_out(p3_rdata));

    // --- PIPELINE STAGE 1: SNN Layer 1 ---
    SNN_Layer_Refactored_Streaming #( .IN_NEURONS(SNN_L1_IN), .OUT_NEURONS(SNN_L1_OUT) )
    SNN1 (
        .clk(clk), .rst_n(rst_n), .i_clk_enable(inference_busy),
        .i_spike_valid(i_cnn_spike_valid), .i_spike_time(i_cnn_spike_time),
        .i_spike_addr(i_cnn_spike_addr), .i_last_spike(i_cnn_last_pixel_sent),
        .o_spike_ack(o_cnn_spike_ack),
        .o_result_valid(s1_out_valid), .o_result_data(s1_out_data),
        .o_last_result(s1_out_last), .i_result_ack(s1_out_ack),
        .weight_ram_addr(w1_addr), .weight_ram_rdata(w1_rdata),
        .param_ram_addr(p1_addr), .param_ram_rdata(p1_rdata),
        .o_done(snn1_done)
    );

    // --- PIPELINE STAGE 2: Bridge L1->L2 ---
    AER_Bridge_Refactored_Streaming #( .NUM_INPUTS(SNN_L1_OUT) )
    Bridge2 (
        .clk(clk), .rst_n(rst_n), .i_clk_enable(inference_busy),
        .i_result_valid(s1_out_valid), .i_result_data(s1_out_data),
        .i_last_result(s1_out_last), .o_result_ack(s1_out_ack),
        .o_req(b2_out_req), .i_ack(b2_out_ack), .o_req_type(b2_out_req_type),
        .o_spike_time(b2_out_spike_time), .o_spike_addr(b2_out_spike_addr),
        .o_done(bridge2_done)
    );
    
    // --- PIPELINE STAGE 3: SNN Layer 2 ---
    SNN_Layer_Refactored_Streaming #( .IN_NEURONS(SNN_L2_IN), .OUT_NEURONS(SNN_L2_OUT) )
    SNN2 (
        .clk(clk), .rst_n(rst_n), .i_clk_enable(inference_busy),
        .i_spike_valid(b2_out_req), .i_spike_time(b2_out_spike_time),
        .i_spike_addr(b2_out_spike_addr), .i_last_spike(b2_out_req_type),
        .o_spike_ack(b2_out_ack),
        .o_result_valid(s2_out_valid), .o_result_data(s2_out_data),
        .o_last_result(s2_out_last), .i_result_ack(s2_out_ack),
        .weight_ram_addr(w2_addr), .weight_ram_rdata(w2_rdata),
        .param_ram_addr(p2_addr), .param_ram_rdata(p2_rdata),
        .o_done(snn2_done)
    );

    // --- PIPELINE STAGE 4: Bridge L2->L3 ---
    AER_Bridge_Refactored_Streaming #( .NUM_INPUTS(SNN_L2_OUT) )
    Bridge3 (
        .clk(clk), .rst_n(rst_n), .i_clk_enable(inference_busy),
        .i_result_valid(s2_out_valid), .i_result_data(s2_out_data),
        .i_last_result(s2_out_last), .o_result_ack(s2_out_ack),
        .o_req(b3_out_req), .i_ack(b3_out_ack), .o_req_type(b3_out_req_type),
        .o_spike_time(b3_out_spike_time), .o_spike_addr(b3_out_spike_addr),
        .o_done(bridge3_done)
    );

    // --- PIPELINE STAGE 5: SNN Layer 3 ---
    SNN_Layer_Refactored_Streaming #( .IS_OUTPUT_LAYER(1), .IN_NEURONS(SNN_L3_IN), .OUT_NEURONS(SNN_L3_OUT) )
    SNN3 (
        .clk(clk), .rst_n(rst_n), .i_clk_enable(inference_busy),
        .i_spike_valid(b3_out_req), .i_spike_time(b3_out_spike_time),
        .i_spike_addr(b3_out_spike_addr), .i_last_spike(b3_out_req_type),
        .o_spike_ack(b3_out_ack),
        .o_result_valid(s3_out_valid), .o_result_data(s3_out_data),
        .o_last_result(s3_out_last), .i_result_ack(s3_out_ack),
        .weight_ram_addr(w3_addr), .weight_ram_rdata(w3_rdata),
        .param_ram_addr(p3_addr), .param_ram_rdata(p3_rdata),
        .o_done(snn3_done)
    );

    // --- PIPELINE STAGE 6: Stream to Parallel Adapter ---
    Stream_To_Parallel #( .VEC_LEN(SNN_L3_OUT), .DATA_W(ACC_W) )
    S2P_Adapter (
        .clk(clk), .rst_n(rst_n),
        .i_valid(s3_out_valid), .i_data(s3_out_data), .i_last(s3_out_last), .o_ack(s3_out_ack),
        .o_data_flat(s2p_data_flat), .o_valid_out(s2p_valid_out)
    );

    // --- PIPELINE STAGE 7: Final Decision ---
    ArgMax_Unit_Refactored #( .VEC_LEN(SNN_L3_OUT), .DATA_W(ACC_W) )
    ArgMax (
        .clk(clk), .rst_n(rst_n), .i_clk_enable(inference_busy),
        .i_start(s2p_valid_out),
        .o_done(argmax_done),
        .i_potentials_flat(s2p_data_flat),
        .o_predicted_class(o_predicted_class)
    );
    
    assign o_inference_done = argmax_done;
    
endmodule