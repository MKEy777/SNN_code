`timescale 1ns / 1ps

module AER_Bridge_Refactored_Streaming #(
    parameter NUM_INPUTS = 64,
    parameter DATA_W     = 32,
    parameter signed [DATA_W-1:0] T_MAX = 32'h7FFFFFFF
) (
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire                      i_clk_enable,

    // Upstream Interface (Data Input)
    input  wire                      i_result_valid,
    input  wire signed [DATA_W-1:0]  i_result_data,
    input  wire                      i_last_result,
    output wire                      o_result_ack,

    // Downstream Interface (Spike Output)
    output reg                       o_req,
    input  wire                      i_ack,
    output reg                       o_req_type,
    output reg signed [DATA_W-1:0]   o_spike_time,
    output reg [$clog2(NUM_INPUTS)-1:0] o_spike_addr,

    // Status
    output reg                       o_done
);
    
    localparam S_IDLE       = 3'd0;
    localparam S_LOAD       = 3'd1;
    localparam S_SEND_LOOP  = 3'd2;
    localparam S_FINALIZE   = 3'd3;
    localparam S_DONE_PULSE = 3'd4;
    reg [2:0] state, next_state;

    reg signed [DATA_W-1:0] spike_ram [0:NUM_INPUTS-1];
    reg [$clog2(NUM_INPUTS)-1:0] load_cntr;
    reg [$clog2(NUM_INPUTS)-1:0] send_cntr;
    
    assign o_result_ack = (state == S_LOAD);

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) state <= S_IDLE;
        else if (i_clk_enable) state <= next_state;
    end

    always @(*) begin
        next_state = state;
        o_req = 1'b0;
        o_req_type = 1'b0;
        o_spike_time = 'd0;
        o_spike_addr = 'd0;
        o_done = 1'b0;

        case(state)
            S_IDLE:       if(i_result_valid) next_state = S_LOAD;
            S_LOAD:       if(i_result_valid && i_last_result) next_state = S_SEND_LOOP;
            S_SEND_LOOP:  begin
                o_req = (spike_ram[send_cntr] < T_MAX);
                o_spike_time = spike_ram[send_cntr];
                o_spike_addr = send_cntr;
                if ((o_req && i_ack) || !o_req) begin
                    if (send_cntr == NUM_INPUTS - 1) next_state = S_FINALIZE;
                end
            end
            S_FINALIZE: begin
                o_req = 1'b1;
                o_req_type = 1'b1;
                if(i_ack) next_state = S_DONE_PULSE;
            end
            S_DONE_PULSE: begin
                o_done = 1'b1;
                next_state = S_IDLE;
            end
            default: next_state = S_IDLE;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            load_cntr <= 0;
            send_cntr <= 0;
        end else if (i_clk_enable) begin
            if (state == S_IDLE && next_state == S_LOAD) begin
                load_cntr <= 0;
            end else if (state == S_LOAD && i_result_valid) begin
                spike_ram[load_cntr] <= i_result_data;
                if (load_cntr < NUM_INPUTS - 1) load_cntr <= load_cntr + 1;
            end
            
            if (next_state == S_SEND_LOOP && state != S_SEND_LOOP) begin
                send_cntr <= 0;
            end else if (state == S_SEND_LOOP) begin
                if ((o_req && i_ack) || !o_req) begin
                    if (send_cntr < NUM_INPUTS - 1) send_cntr <= send_cntr + 1;
                end
            end
        end
    end

endmodule