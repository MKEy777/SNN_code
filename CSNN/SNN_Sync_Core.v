`timescale 1ns / 1ps

module SNN_Sync_Core #(
    parameter IN_NEURONS      = 288,
    parameter OUT_NEURONS     = 64,
    parameter TIME_W          = 32,
    parameter WEIGHT_W        = 8,
    parameter ACC_W           = 48,
    parameter IS_OUTPUT_LAYER = 0, 
    
    parameter signed [TIME_W-1:0] T_MAX        = 32'h7FFFFFFF,
    parameter signed [TIME_W-1:0] T_MIN        = 32'h00010000,
    parameter signed [TIME_W-1:0] T_MIN_PREV   = 32'h00000000
) (
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire                      i_start_computation,
    output reg                       o_computation_done,
    input  wire                      i_finalize_sample,

    // --- Data Inputs ---
    input  wire signed [TIME_W-1:0]  i_spike_time,
    input  wire [$clog2(IN_NEURONS)-1:0] i_spike_addr,
    
    // --- Weight Memory Interface ---
    output reg  [$clog2(IN_NEURONS*OUT_NEURONS)-1:0] weight_ram_addr,
    input  wire signed [WEIGHT_W-1:0]                 weight_ram_rdata,

    // --- Neuron State Memory (Vm) Interface ---
    output reg                                       neuron_ram_wen,
    output reg  [$clog2(OUT_NEURONS)-1:0]            neuron_ram_addr,
    output reg  signed [ACC_W-1:0]                   neuron_ram_wdata,
    input  wire signed [ACC_W-1:0]                   neuron_ram_rdata,

    // --- Generic Parameter ROM Interface (for D_i or Bias) ---
    output reg [$clog2(OUT_NEURONS)-1:0]             param_ram_addr,
    input wire signed [ACC_W-1:0]                    param_ram_rdata,

    // --- Final Result Output ---
    output reg signed [OUT_NEURONS*ACC_W-1:0] o_result_flat
);

    // --- FSM states ---
    localparam S_IDLE        = 3'b000;
    localparam S_ACCUM_READ  = 3'b001; 
    localparam S_ACCUM_WRITE = 3'b100; 
    localparam S_FINALIZE    = 3'b010;
    localparam S_DONE        = 3'b011;
    
    reg [2:0] state, next_state;

    reg signed [TIME_W-1:0] latched_spike_time;
    reg [$clog2(IN_NEURONS)-1:0] latched_spike_addr;
    reg [$clog2(OUT_NEURONS)-1:0] update_cntr;
    
    wire signed [TIME_W-1:0] time_diff;
    assign time_diff = (IS_OUTPUT_LAYER == 1) ? (T_MIN - latched_spike_time) : (latched_spike_time - T_MIN);
                       
    // Combinational Logic for FSM
    always @(*) begin
        next_state = state;
        o_computation_done = 1'b0;
        neuron_ram_wen = 1'b0;
        neuron_ram_wdata = {ACC_W{1'bx}};
        weight_ram_addr = {($clog2(IN_NEURONS*OUT_NEURONS)){1'bx}};
        neuron_ram_addr = {($clog2(OUT_NEURONS)){1'bx}};
        param_ram_addr = {($clog2(OUT_NEURONS)){1'bx}};

        case (state)
            S_IDLE: begin
                if (i_start_computation)      next_state = S_ACCUM_READ; 
                else if (i_finalize_sample) next_state = S_FINALIZE;
            end
            
            S_ACCUM_READ: begin
                weight_ram_addr = latched_spike_addr * OUT_NEURONS + update_cntr;
                neuron_ram_addr = update_cntr;
                next_state = S_ACCUM_WRITE; 
            end

            S_ACCUM_WRITE: begin
                weight_ram_addr = 0; 
                neuron_ram_addr = 0; 
                neuron_ram_wen = 1'b1;
                neuron_ram_wdata = neuron_ram_rdata + ($signed(time_diff) * $signed(weight_ram_rdata));
                
                if (update_cntr == OUT_NEURONS - 1) begin
                    next_state = S_DONE;
                end else begin
                    next_state = S_ACCUM_READ; 
                end
            end

            S_FINALIZE: begin
                neuron_ram_addr = update_cntr;
                param_ram_addr = update_cntr;
                weight_ram_addr = 0; 
                if (update_cntr == OUT_NEURONS - 1) next_state = S_DONE;
                else                                next_state = S_FINALIZE;
            end
            
            S_DONE: begin
                o_computation_done = 1'b1;
                next_state = S_IDLE;
            end
            
            default: next_state = S_IDLE;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            update_cntr <= 0;
            latched_spike_time <= 0;
            latched_spike_addr <= 0;
            o_result_flat <= 0;
        end else begin
            state <= next_state;
            
            if (state == S_IDLE && i_start_computation) begin
                latched_spike_time <= i_spike_time;
                latched_spike_addr <= i_spike_addr;
            end

            if (state == S_ACCUM_WRITE || state == S_FINALIZE) begin
                if (update_cntr == OUT_NEURONS - 1) begin
                    update_cntr <= 0;
                end else begin
                    update_cntr <= update_cntr + 1;
                end
            end else if (next_state == S_ACCUM_READ && state != S_ACCUM_WRITE) begin
                 update_cntr <= 0;
            end
            
            if (state == S_FINALIZE) begin
                if (IS_OUTPUT_LAYER == 0) begin 
                    o_result_flat[update_cntr*ACC_W +: ACC_W] <= neuron_ram_rdata + T_MAX - param_ram_rdata;
                end else begin 
                    o_result_flat[update_cntr*ACC_W +: ACC_W] <= neuron_ram_rdata + param_ram_rdata;
                end
            end
        end
    end

endmodule