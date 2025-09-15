`timescale 1ns / 1ps

module ArgMax_Unit_Refactored #(
    parameter VEC_LEN = 3,  // Vector length (number of classes)
    parameter DATA_W  = 48  // Data width of the potential vector
) (
    // --- System & Control ---
    input wire                      clk,
    input wire                      rst_n,
    input wire                      i_clk_enable,       // New: For clock gating

    // --- High-Level Control (from snn_top FSM) ---
    input wire                      i_start,            // New: Level-sensitive start signal
    output reg                      o_done,             // New: Pulsed when calculation is complete

    // --- Data Input ---
    input wire signed [VEC_LEN*DATA_W-1:0] i_potentials_flat,

    // --- Data Output ---
    output reg [$clog2(VEC_LEN)-1:0]  o_predicted_class
);

    // --- FSM States ---
    localparam S_IDLE          = 2'b00;
    localparam S_CALC_AND_REG  = 2'b01;
    localparam S_DONE_PULSE    = 2'b10;

    reg [1:0] state, next_state;

    reg signed [DATA_W-1:0] max_val;
    reg [$clog2(VEC_LEN)-1:0] max_idx;
    integer i;
    
    always @(*) begin
        max_val = i_potentials_flat[DATA_W-1:0];
        max_idx = 0;
    
        // Loop to find the maximum
        for (i = 1; i < VEC_LEN; i = i + 1) begin
            if (i_potentials_flat[(i+1)*DATA_W-1 -: DATA_W] > max_val) begin
                max_val = i_potentials_flat[(i+1)*DATA_W-1 -: DATA_W];
                max_idx = i;
            end
        end
    end


    // --- FSM State Register ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
        end else if (i_clk_enable) begin // Clock Gating
            state <= next_state;
        end
    end
    
    // --- FSM Combinational Logic & Sequential Output Registering ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_done <= 1'b0;
            o_predicted_class <= 0;
        end else if (i_clk_enable) begin // Clock Gating
            // Default assignments
            o_done <= 1'b0;
            
            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        o_predicted_class <= max_idx;
                    end
                end
                
                S_CALC_AND_REG: begin

                end

                S_DONE_PULSE: begin
                    // Assert o_done for one cycle
                    o_done <= 1'b1;
                end
            endcase
        end
    end
    
    // FSM next state logic
     always @(*) begin
        next_state = state;
        case(state)
            S_IDLE:         if (i_start) next_state = S_CALC_AND_REG;
            S_CALC_AND_REG: next_state = S_DONE_PULSE;
            S_DONE_PULSE:   next_state = S_IDLE;
            default:        next_state = S_IDLE;
        endcase
    end

endmodule