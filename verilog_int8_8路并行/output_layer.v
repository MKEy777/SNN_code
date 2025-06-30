// 文件名: output_layer_3stage_quant_final.v
// 模块: output_layer (最终版 - 适配3级量化单元)
`timescale 1ns / 1ps

module output_layer #(
    parameter integer PARALLEL_FACTOR = 8, INPUT_SIZE = 128, OUTPUT_SIZE = 3, INPUT_DATA_WIDTH = 8,
    parameter OUTPUT_DATA_WIDTH = 16, ACC_WIDTH = 32, BIAS_WIDTH = 32,
    parameter WEIGHT_ROM_FILE = "output_w_rom.mem", parameter BIAS_ROM_FILE = "output_b_rom.mem",
    parameter integer LOGITS_SCALE_SHIFT = 8, parameter signed [8:0] INPUT_ZERO_POINT = 0
)(
    input  wire clk, rst_n, i_valid, o_ready,
    output wire i_ready, o_valid,
    input  wire signed [INPUT_SIZE*INPUT_DATA_WIDTH-1:0]   i_spikes,
    output wire signed [OUTPUT_SIZE*OUTPUT_DATA_WIDTH-1:0] o_logits
);
    integer j;

    localparam FSM_IDLE         = 4'd0, FSM_LOAD = 4'd1, FSM_CALC_SETUP = 4'd2,
               FSM_CALC_START   = 4'd3, FSM_WAIT_ENGINE = 4'd4, FSM_QUANT_START  = 4'd5, 
               FSM_WAIT_QUANT   = 4'd6, FSM_DONE = 4'd7;
    reg [3:0] current_state, next_state;

    // ... 其余寄存器和线网定义不变 ...
    localparam WORDS_PER_NEURON = (INPUT_SIZE + PARALLEL_FACTOR - 1) / PARALLEL_FACTOR;
    localparam ROM_ADDR_WIDTH   = clog2(OUTPUT_SIZE * WORDS_PER_NEURON);
    reg [$clog2(INPUT_SIZE):0]    input_cnt;
    reg [$clog2(OUTPUT_SIZE)-1:0] output_cnt;
    reg signed [INPUT_DATA_WIDTH-1:0] i_spikes_mem [0:INPUT_SIZE-1];
    reg signed [OUTPUT_SIZE*OUTPUT_DATA_WIDTH-1:0] o_logits_reg;
    reg signed [ACC_WIDTH-1:0]    accumulator_full;
    reg                          is_first_accumulation;
    reg signed [INPUT_DATA_WIDTH*PARALLEL_FACTOR-1:0] parallel_input_data_reg;
    wire signed [BIAS_WIDTH-1:0]  bias_data;
    wire signed [INPUT_DATA_WIDTH*PARALLEL_FACTOR-1:0] parallel_weight_data;
    wire signed [INPUT_DATA_WIDTH*PARALLEL_FACTOR-1:0] parallel_input_data;
    wire signed [ACC_WIDTH-1:0]   sum_of_products;
    wire signed [OUTPUT_DATA_WIDTH-1:0] clamped_output;
    wire is_last_mac_cycle = (input_cnt >= INPUT_SIZE - PARALLEL_FACTOR);
    wire is_last_neuron    = (output_cnt == OUTPUT_SIZE - 1);
    wire engine_i_valid;
    wire engine_o_valid;
    wire quant_i_valid;
    wire quant_o_valid;

    assign i_ready  = (current_state == FSM_IDLE);
    assign o_valid  = (current_state == FSM_DONE);
    assign o_logits = o_logits_reg;
    assign engine_i_valid = (current_state == FSM_CALC_START);
    assign quant_i_valid  = (current_state == FSM_QUANT_START);

    // 模块例化: 使用3级流水线的quantization_unit
    parallel_compute_engine #(.PARALLEL_FACTOR(PARALLEL_FACTOR), .DATA_WIDTH(INPUT_DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH), .INPUT_ZERO_POINT(INPUT_ZERO_POINT))
        u_compute_engine (.clk(clk), .rst_n(rst_n), .i_valid(engine_i_valid), .o_valid(engine_o_valid), .parallel_inputs(parallel_input_data_reg), .parallel_weights(parallel_weight_data), .sum_of_products(sum_of_products));
    quantization_unit #(.DATA_WIDTH(OUTPUT_DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH), .BIAS_WIDTH(BIAS_WIDTH), .OUTPUT_MULTIPLIER(32'sd1), .OUTPUT_ZERO_POINT(32'sd0), .OUTPUT_SHIFT(LOGITS_SCALE_SHIFT))
        u_quant_unit (.clk(clk), .rst_n(rst_n), .i_valid(quant_i_valid), .o_valid(quant_o_valid), .accumulator_in(accumulator_full), .bias_in(bias_data), .clamped_output(clamped_output));
    rom_sync #( .DATA_WIDTH (INPUT_DATA_WIDTH * PARALLEL_FACTOR), .ADDR_WIDTH (ROM_ADDR_WIDTH), .MEM_FILE (WEIGHT_ROM_FILE) ) 
        weight_rom ( .clk(clk), .addr(output_cnt * WORDS_PER_NEURON + (input_cnt / PARALLEL_FACTOR)), .data_out(parallel_weight_data) );
    rom_sync #( .DATA_WIDTH (BIAS_WIDTH), .ADDR_WIDTH (clog2(OUTPUT_SIZE)), .MEM_FILE (BIAS_ROM_FILE) ) 
        bias_rom ( .clk(clk), .addr(output_cnt), .data_out(bias_data) );

    genvar k;
    generate for (k = 0; k < PARALLEL_FACTOR; k = k + 1) begin
        assign parallel_input_data[k*INPUT_DATA_WIDTH +: INPUT_DATA_WIDTH] = ((input_cnt + k) < INPUT_SIZE) ? i_spikes_mem[input_cnt + k] : 0;
    end endgenerate

    // FSM 状态转移逻辑 (最终版)
    always @(current_state or i_valid or engine_o_valid or quant_o_valid or is_last_mac_cycle or is_last_neuron or o_ready) begin
        next_state = current_state;
        case (current_state)
            FSM_IDLE:         if (i_valid) next_state = FSM_LOAD;
            FSM_LOAD:         next_state = FSM_CALC_SETUP;
            FSM_CALC_SETUP:   next_state = FSM_CALC_START;
            FSM_CALC_START:   next_state = FSM_WAIT_ENGINE;
            FSM_WAIT_ENGINE:  if (engine_o_valid) begin
                                 if (is_last_mac_cycle) next_state = FSM_QUANT_START;
                                 else next_state = FSM_CALC_SETUP;
                              end
            FSM_QUANT_START:  next_state = FSM_WAIT_QUANT;
            FSM_WAIT_QUANT:   if (quant_o_valid) begin
                                 if (is_last_neuron) next_state = FSM_DONE;
                                 else next_state = FSM_CALC_SETUP;
                              end
            FSM_DONE:         if (o_ready) next_state = FSM_IDLE;
        endcase
    end

    // FSM 时序与数据处理逻辑 (最终版)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= FSM_IDLE; input_cnt <= 0; output_cnt <= 0;
            accumulator_full <= 0; o_logits_reg <= 0;
            is_first_accumulation <= 1'b0; parallel_input_data_reg <= 0;
        end else begin
            current_state <= next_state;
            parallel_input_data_reg <= parallel_input_data;

            case (current_state)
                FSM_IDLE: if (i_valid) begin
                    input_cnt <= 0; output_cnt <= 0; accumulator_full <= 0; is_first_accumulation <= 1'b1;
                end
                FSM_LOAD: for (j = 0; j < INPUT_SIZE; j = j + 1) begin
                    i_spikes_mem[j] <= i_spikes[j*INPUT_DATA_WIDTH +: INPUT_DATA_WIDTH];
                end
                FSM_CALC_SETUP: begin
                end 
                FSM_CALC_START:  begin
                end 
                FSM_WAIT_ENGINE: if (engine_o_valid) begin
                    if (is_first_accumulation) begin
                        accumulator_full <= sum_of_products; is_first_accumulation <= 1'b0;
                    end else begin
                        accumulator_full <= accumulator_full + sum_of_products;
                    end
                    input_cnt <= input_cnt + PARALLEL_FACTOR;
                end
                FSM_QUANT_START: begin
                    // 仅用于产生quant_i_valid脉冲
                end
                FSM_WAIT_QUANT: if (quant_o_valid) begin
                    o_logits_reg[output_cnt*OUTPUT_DATA_WIDTH +: OUTPUT_DATA_WIDTH] <= clamped_output;
                    if (!is_last_neuron) begin
                        output_cnt <= output_cnt + 1; input_cnt <= 0; accumulator_full <= 0; is_first_accumulation <= 1'b1;
                    end
                end
                FSM_DONE: if (o_ready) begin
                    input_cnt <= 0; output_cnt <= 0;
                end
            endcase
        end
    end
    
    function integer clog2;
        input integer value;
        begin value = value - 1; for (clog2 = 0; value > 0; clog2 = clog2 + 1) value = value >> 1; end
    endfunction
endmodule