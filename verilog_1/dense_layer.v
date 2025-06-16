`timescale 1ns / 1ps

module dense_layer #(
    parameter INPUT_SIZE        = 300,
    parameter OUTPUT_SIZE       = 256,
    parameter DATA_WIDTH        = 32,
    parameter FRACTIONAL_BITS   = 16,
    parameter WEIGHT_ROM_FILE   = "layer_w_rom.mem",
    parameter BIAS_ROM_FILE     = "layer_d_rom.mem",
    parameter signed [DATA_WIDTH-1:0] T_MIN = 32'h00010000,
    parameter signed [DATA_WIDTH-1:0] T_MAX = 32'h00020000
)(
    input  wire                           clk,
    input  wire                           rst_n,
    input  wire                           i_valid,
    output wire                           i_ready,
    output wire                           o_valid,
    input  wire                           o_ready,
    input  wire signed [INPUT_SIZE*DATA_WIDTH-1:0]   i_spikes,
    output wire signed [OUTPUT_SIZE*DATA_WIDTH-1:0]  o_spikes
);

localparam FSM_IDLE   = 2'd0;
localparam FSM_CALC   = 2'd1;
localparam FSM_FINISH = 2'd2;
localparam FSM_DONE   = 2'd3;
reg [1:0] current_state, next_state;
reg [$clog2(INPUT_SIZE)-1:0]  input_cnt;
reg [$clog2(OUTPUT_SIZE)-1:0] output_cnt;
reg signed [INPUT_SIZE*DATA_WIDTH-1:0]  i_spikes_reg;
reg signed [OUTPUT_SIZE*DATA_WIDTH-1:0] o_spikes_reg;
localparam FULL_ACC_WIDTH = 2 * DATA_WIDTH + $clog2(INPUT_SIZE);
reg signed [FULL_ACC_WIDTH-1:0] accumulator_full;

reg signed [DATA_WIDTH-1:0] tj_shifted_p1;
wire signed [DATA_WIDTH-1:0] weight_data;
wire signed [DATA_WIDTH-1:0] bias_data; // 此处的 bias_data 现在读取的是 "combined_bias.mem"

wire is_last_mac_cycle  = (input_cnt == INPUT_SIZE - 1);
wire is_last_neuron     = (output_cnt == OUTPUT_SIZE - 1);
reg  is_last_mac_cycle_d1;
wire signed [DATA_WIDTH-1:0] current_tj = i_spikes_reg[input_cnt*DATA_WIDTH +: DATA_WIDTH];
wire signed [DATA_WIDTH-1:0] tj_shifted = current_tj - T_MIN;
wire signed [2*DATA_WIDTH-1:0] product = tj_shifted_p1 * weight_data;
wire signed [FULL_ACC_WIDTH-1:0] t_max_aligned  = $signed(T_MAX) <<< FRACTIONAL_BITS;
wire signed [FULL_ACC_WIDTH-1:0] bias_aligned   = $signed(bias_data) <<< FRACTIONAL_BITS;

// ========================[ 关键修改 ]========================
// 原代码: wire signed [FULL_ACC_WIDTH-1:0] sum_full_precision = accumulator_full + (t_max_aligned - bias_aligned);
// 修改后: 直接将累加器与 "组合式偏置" 相加，移除了硬件减法器
wire signed [FULL_ACC_WIDTH-1:0] sum_full_precision = accumulator_full + bias_aligned;
// ==========================================================

wire signed [FULL_ACC_WIDTH-1:0] clipped_result = (sum_full_precision > t_max_aligned) ? t_max_aligned : sum_full_precision;
wire signed [DATA_WIDTH-1:0] ti_final = $signed(clipped_result >>> FRACTIONAL_BITS);
rom_sync #( .DATA_WIDTH (DATA_WIDTH), .ADDR_WIDTH ($clog2(INPUT_SIZE * OUTPUT_SIZE)), .MEM_FILE (WEIGHT_ROM_FILE) ) weight_rom ( .clk(clk), .addr(output_cnt * INPUT_SIZE + input_cnt), .data_out(weight_data) );
rom_sync #( .DATA_WIDTH (DATA_WIDTH), .ADDR_WIDTH ($clog2(OUTPUT_SIZE)), .MEM_FILE (BIAS_ROM_FILE) ) bias_rom ( .clk(clk), .addr(output_cnt), .data_out(bias_data) );
always @(*) begin
    next_state = current_state;
    case (current_state)
        FSM_IDLE:   if (i_valid) next_state = FSM_CALC;
        FSM_CALC:   if (is_last_mac_cycle_d1) next_state = FSM_FINISH;
        FSM_FINISH: if (is_last_neuron) next_state = FSM_DONE;
                    else next_state = FSM_CALC;
        FSM_DONE:   if (o_ready) next_state = FSM_IDLE;
    endcase
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        current_state        <= FSM_IDLE;
        input_cnt            <= 0;
        output_cnt           <= 0;
        accumulator_full     <= 0;
        tj_shifted_p1        <= 0;
        o_spikes_reg         <= 0;
        is_last_mac_cycle_d1 <= 1'b0;
    end else begin
        current_state        <= next_state;
        is_last_mac_cycle_d1 <= is_last_mac_cycle;
        tj_shifted_p1        <= tj_shifted;
        case (current_state)
            FSM_IDLE: begin
                if (i_valid) begin
                    i_spikes_reg     <= i_spikes;
                    input_cnt        <= 0;
                    output_cnt       <= 0;
                    accumulator_full <= 0;
                end
            end

            FSM_CALC: begin
                if (input_cnt == 1) begin
                    accumulator_full <= $signed(product);
                end else if (input_cnt > 1) begin
                    accumulator_full <= accumulator_full + $signed(product);
                end
                input_cnt <= input_cnt + 1;
            end

            FSM_FINISH: begin
                o_spikes_reg[output_cnt*DATA_WIDTH +: DATA_WIDTH] <= ti_final;
                if (!is_last_neuron) begin
                    output_cnt       <= output_cnt + 1;
                    input_cnt        <= 0;
                    accumulator_full <= 0;
                end
            end
            
            FSM_DONE: begin end
        endcase
    end
end

assign i_ready  = (current_state == FSM_IDLE);
assign o_valid  = (current_state == FSM_DONE);
assign o_spikes = o_spikes_reg;

endmodule