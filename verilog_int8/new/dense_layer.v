`timescale 1ns / 1ps

//
// 模块: dense_layer (已修正 timing bug)
// 功能: 一个参数化的、流水线化的全连接层硬件实现。
//
module dense_layer #(
    parameter INPUT_SIZE        = 300,
    parameter OUTPUT_SIZE       = 256,
    parameter DATA_WIDTH        = 8,
    parameter ACC_WIDTH         = 32,
    parameter BIAS_WIDTH        = 32,
    parameter WEIGHT_ROM_FILE   = "layer_w_rom.mem",
    parameter BIAS_ROM_FILE     = "layer_b_rom.mem",
    parameter signed [8:0]  INPUT_ZERO_POINT   = 0,
    parameter signed [8:0]  WEIGHT_ZERO_POINT  = 0,
    parameter signed [8:0]  OUTPUT_ZERO_POINT  = 0,
    parameter signed [31:0] OUTPUT_MULTIPLIER  = 0,
    parameter integer       OUTPUT_SHIFT       = 0
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
    // --- 状态机定义 ---
    localparam FSM_IDLE   = 2'd0, FSM_CALC   = 2'd1, FSM_FINISH = 2'd2, FSM_DONE   = 2'd3;
    reg [1:0] current_state, next_state;

    // --- 内部寄存器与连线 ---
    reg [$clog2(INPUT_SIZE):0]    input_cnt;
    reg [$clog2(OUTPUT_SIZE)-1:0] output_cnt;
    reg signed [INPUT_SIZE*DATA_WIDTH-1:0]  i_spikes_reg;
    reg signed [OUTPUT_SIZE*DATA_WIDTH-1:0] o_spikes_reg;
    reg signed [ACC_WIDTH-1:0]  accumulator_full;
    wire signed [DATA_WIDTH-1:0] weight_data;
    wire signed [BIAS_WIDTH-1:0] bias_data;
    reg signed [9:0] input_minus_zp_dly;
    
    wire signed [DATA_WIDTH-1:0] current_input_val = i_spikes_reg[input_cnt*DATA_WIDTH +: DATA_WIDTH];
    wire signed [9:0] sub_result = $signed(current_input_val) - $signed(INPUT_ZERO_POINT);
    wire signed [17:0] product = input_minus_zp_dly * $signed(weight_data);
    
    // FSM 跳转条件
    wire is_last_mac_cycle  = (input_cnt == INPUT_SIZE - 1);
    wire is_last_neuron     = (output_cnt == OUTPUT_SIZE - 1);
    
    // --- 再量化逻辑 ---

    // **【代码修正】**: 创建一个线网，通过组合逻辑计算出真正的最终累加和。
    // 它等于FINISH状态前一刻的累加器值，加上流水线中最后一个乘积。
    wire signed [ACC_WIDTH-1:0] final_acc_val = accumulator_full + $signed(product);

    // **【代码修正】**: 修改 acc_with_bias 的数据来源。
    // 在FINISH状态，它必须使用包含了最后一项乘积的 final_acc_val。
    // 在其他状态，它使用寄存器中的值 (虽然此时其值无效，但在FINISH前不会被使用)。
    wire signed [ACC_WIDTH-1:0]   acc_with_bias = ((current_state == FSM_FINISH) ? final_acc_val : accumulator_full) + $signed(bias_data);
    
    wire signed [ACC_WIDTH+31:0]  multiplied_val, shifted_val_full, val_with_zp_full;
    wire signed [DATA_WIDTH-1:0]  clamped_output;
    assign multiplied_val      = acc_with_bias * $signed(OUTPUT_MULTIPLIER);
    assign shifted_val_full    = $signed(multiplied_val) >>> OUTPUT_SHIFT; 
    assign val_with_zp_full    = shifted_val_full + $signed(OUTPUT_ZERO_POINT); 
    assign clamped_output      = (val_with_zp_full > 127) ? 127 : (val_with_zp_full < -128) ? -128 : val_with_zp_full[DATA_WIDTH-1:0];

    // --- ROM 实例化 ---
    rom_sync #( .DATA_WIDTH (DATA_WIDTH), .ADDR_WIDTH ($clog2(INPUT_SIZE * OUTPUT_SIZE)), .MEM_FILE (WEIGHT_ROM_FILE) ) weight_rom ( .clk(clk), .addr(output_cnt * INPUT_SIZE + input_cnt), .data_out(weight_data) );
    rom_sync #( .DATA_WIDTH (BIAS_WIDTH), .ADDR_WIDTH ($clog2(OUTPUT_SIZE)), .MEM_FILE (BIAS_ROM_FILE) ) bias_rom ( .clk(clk), .addr(output_cnt), .data_out(bias_data) );

    // --- FSM 状态转移逻辑 (无变化) ---
    always @(*) begin
        next_state = current_state;
        case (current_state)
            FSM_IDLE:   if (i_valid) next_state = FSM_CALC;
            FSM_CALC:   if (is_last_mac_cycle) next_state = FSM_FINISH;
            FSM_FINISH: if (is_last_neuron) next_state = FSM_DONE;
                        else next_state = FSM_CALC;
            FSM_DONE:   if (o_ready) next_state = FSM_IDLE;
        endcase
    end

    // --- FSM 时序逻辑 ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= FSM_IDLE; input_cnt <= 0; output_cnt <= 0; accumulator_full <= 0; o_spikes_reg <= 0; input_minus_zp_dly <= 0;
        end else begin
            current_state <= next_state;

            case (current_state)
                FSM_IDLE:
                    if (i_valid) begin
                        i_spikes_reg     <= i_spikes;
                        input_cnt        <= 0;
                        output_cnt       <= 0;
                        accumulator_full <= 0; 
                    end
                FSM_CALC: begin
                    input_minus_zp_dly <= sub_result;
                    
                    // 第一个有效product在input_cnt=1时产生。
                    // 只在 product 有效时累加。
                    if (input_cnt > 0) begin
                        accumulator_full <= accumulator_full + $signed(product);
                    end
                    
                    input_cnt <= input_cnt + 1;
                end
                FSM_FINISH: begin
                    // **【代码修正】**: 移除此处画蛇添足的累加器更新指令。
                    // 最终累加和已通过上面的组合逻辑送去量化，无需在此处更新寄存器。
                    
                    o_spikes_reg[output_cnt*DATA_WIDTH +: DATA_WIDTH] <= clamped_output;
                    
                    if (!is_last_neuron) begin
                        output_cnt       <= output_cnt + 1;
                        input_cnt        <= 0;
                        accumulator_full <= 0; // 为下一个神经元正确清零
                    end
                end
                FSM_DONE: begin
                end
            endcase
        end
    end
    
    assign i_ready  = (current_state == FSM_IDLE);
    assign o_valid  = (current_state == FSM_DONE);
    assign o_spikes = o_spikes_reg;
endmodule