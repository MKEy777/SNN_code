`timescale 1ns / 1ps

module SNN_Core_Streaming #(
    parameter IN_NEURONS      = 64,
    parameter OUT_NEURONS     = 32,
    parameter TIME_W          = 32,
    parameter WEIGHT_W        = 8,
    parameter ACC_W           = 32,
    parameter IS_OUTPUT_LAYER = 0,
    parameter signed [TIME_W-1:0] T_MAX = 32'h7FFFFFFF,
    parameter signed [TIME_W-1:0] T_MIN = 32'h00010000
) (
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire                      i_clk_enable,

    input  wire                      i_spike_valid,
    input  wire signed [TIME_W-1:0]  i_spike_time,
    input  wire [$clog2(IN_NEURONS)-1:0] i_spike_addr,
    input  wire                      i_last_spike,
    output wire                      o_spike_ack,

    output reg                       o_result_valid,
    output reg signed [ACC_W-1:0]    o_result_data,
    output reg                       o_last_result,
    input  wire                      i_result_ack,

    output reg [$clog2(IN_NEURONS*OUT_NEURONS)-1:0] weight_ram_addr,
    input  wire signed [WEIGHT_W-1:0]                 weight_ram_rdata,
    
    output reg [$clog2(OUT_NEURONS)-1:0]             param_ram_addr,
    input  wire signed [ACC_W-1:0]                    param_ram_rdata,

    output reg                       o_layer_done
);

    localparam S_IDLE              = 4'd0;
    localparam S_ACCUM_FETCH_SPIKE = 4'd1;
    localparam S_ACCUM_READ_MEM    = 4'd2;
    localparam S_ACCUM_WRITE_MEM   = 4'd3;
    localparam S_FINALIZE_READ_MEM = 4'd4;
    localparam S_FINALIZE_OUTPUT   = 4'd5;
    localparam S_DONE_PULSE        = 4'd6;

    reg [3:0] state, next_state;

    (* ram_style = "block" *)
    reg signed [ACC_W-1:0] neuron_potentials_ram [0:OUT_NEURONS-1];

    reg [$clog2(OUT_NEURONS)-1:0] neuron_idx;
    reg                           is_last_spike_latched;
    reg                           is_first_spike_of_stream;
    reg signed [TIME_W-1:0]       latched_spike_time;
    reg [$clog2(IN_NEURONS)-1:0]  latched_spike_addr;
    reg signed [ACC_W-1:0]        latched_old_potential;

    wire signed [TIME_W-1:0] time_diff = latched_spike_time - T_MIN;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
        end else if (i_clk_enable) begin
            state <= next_state;
        end
    end

    always @(*) begin
        next_state          = state;
        o_layer_done        = 1'b0;
        o_result_valid      = 1'b0;
        o_last_result       = 1'b0;
        o_result_data       = 'd0;
        weight_ram_addr     = 'd0;
        param_ram_addr      = 'd0;

        case(state)
            S_IDLE: begin
                if (i_spike_valid) begin
                    next_state = S_ACCUM_FETCH_SPIKE;
                end
            end

            S_ACCUM_FETCH_SPIKE: begin
                if (i_spike_valid) begin
                    next_state = S_ACCUM_READ_MEM;
                end else if (is_last_spike_latched) begin
                    next_state = S_FINALIZE_READ_MEM;
                end
            end

            S_ACCUM_READ_MEM: begin
                next_state = S_ACCUM_WRITE_MEM;
            end

            S_ACCUM_WRITE_MEM: begin
                if (neuron_idx == OUT_NEURONS - 1) begin
                    if (is_last_spike_latched) begin
                        next_state = S_FINALIZE_READ_MEM;
                    end else begin
                        next_state = S_ACCUM_FETCH_SPIKE;
                    end
                end else begin
                    next_state = S_ACCUM_READ_MEM;
                end
            end

            S_FINALIZE_READ_MEM: begin
                next_state = S_FINALIZE_OUTPUT;
            end

            S_FINALIZE_OUTPUT: begin
                o_result_valid = 1'b1;
                if (IS_OUTPUT_LAYER == 0) begin
                    o_result_data = latched_old_potential + T_MAX - param_ram_rdata;
                end else begin
                    o_result_data = latched_old_potential + param_ram_rdata;
                end

                if (i_result_ack) begin
                    if (neuron_idx == OUT_NEURONS - 1) begin
                        next_state = S_DONE_PULSE;
                    end else begin
                        next_state = S_FINALIZE_READ_MEM;
                    end
                end else begin
                    next_state = S_FINALIZE_OUTPUT;
                end
            end

            S_DONE_PULSE: begin
                o_layer_done = 1'b1;
                next_state = S_IDLE;
            end

            default: next_state = S_IDLE;
        endcase

        if ((state == S_FINALIZE_OUTPUT) && (neuron_idx == OUT_NEURONS - 1)) begin
            o_last_result = 1'b1;
        end
    end
    
    assign o_spike_ack = (state == S_ACCUM_FETCH_SPIKE || state == S_IDLE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            neuron_idx                 <= 0;
            is_last_spike_latched      <= 1'b0;
            is_first_spike_of_stream   <= 1'b0;
            latched_spike_time         <= 0;
            latched_spike_addr         <= 0;
            latched_old_potential      <= 0;
            for (integer i = 0; i < OUT_NEURONS; i = i + 1) begin
                neuron_potentials_ram[i] <= 0;
            end
        end else if (i_clk_enable) begin
            case (state)
                S_IDLE: begin
                    neuron_idx <= 0;
                    is_last_spike_latched <= 1'b0;
                    if (i_spike_valid) begin
                        is_first_spike_of_stream <= 1'b1;
                    end
                end
                
                S_ACCUM_FETCH_SPIKE: begin
                    if (i_spike_valid) begin
                        latched_spike_time <= i_spike_time;
                        latched_spike_addr <= i_spike_addr;
                        is_last_spike_latched <= i_last_spike;
                        neuron_idx <= 0;
                    end
                end

                S_ACCUM_READ_MEM: begin
                    weight_ram_addr <= latched_spike_addr * OUT_NEURONS + neuron_idx;
                    latched_old_potential <= neuron_potentials_ram[neuron_idx];
                end

                S_ACCUM_WRITE_MEM: begin
                    if (is_first_spike_of_stream) begin
                        neuron_potentials_ram[neuron_idx] <= time_diff * $signed(weight_ram_rdata);
                    end else begin
                        neuron_potentials_ram[neuron_idx] <= latched_old_potential + (time_diff * $signed(weight_ram_rdata));
                    end
                    
                    if (neuron_idx < OUT_NEURONS - 1) begin
                        neuron_idx <= neuron_idx + 1;
                    end else begin
                        is_first_spike_of_stream <= 1'b0;
                    end
                end

                S_FINALIZE_READ_MEM: begin
                    param_ram_addr <= neuron_idx;
                    latched_old_potential <= neuron_potentials_ram[neuron_idx];
                end

                S_FINALIZE_OUTPUT: begin
                    if (i_result_ack && neuron_idx < OUT_NEURONS - 1) begin
                        neuron_idx <= neuron_idx + 1;
                    end
                end
                
                S_DONE_PULSE: begin
                    neuron_idx <= 0;
                    is_last_spike_latched <= 1'b0;
                end
            endcase
        end
    end

endmodule