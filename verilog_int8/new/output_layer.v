`timescale 1ns / 1ps

//
// ģ��: output_layer (������ timing bug)
//
module output_layer #(
    parameter INPUT_SIZE         = 128,
    parameter OUTPUT_SIZE        = 3,
    parameter INPUT_DATA_WIDTH   = 8,
    parameter OUTPUT_DATA_WIDTH  = 16,
    parameter ACC_WIDTH          = 32,
    parameter BIAS_WIDTH         = 32,
    parameter WEIGHT_ROM_FILE    = "output_w_rom.mem",
    parameter BIAS_ROM_FILE      = "output_b_rom.mem",
    parameter integer LOGITS_SCALE_SHIFT = 8
)(
    input  wire                             clk,
    input  wire                             rst_n,
    input  wire                             i_valid,
    output wire                             i_ready,
    output wire                             o_valid,
    input  wire                             o_ready,
    input  wire signed [INPUT_SIZE*INPUT_DATA_WIDTH-1:0]   i_spikes,
    output wire signed [OUTPUT_SIZE*OUTPUT_DATA_WIDTH-1:0] o_logits
);
    localparam FSM_IDLE = 2'd0, FSM_CALC = 2'd1, FSM_FINISH = 2'd2, FSM_DONE = 2'd3;
    reg [1:0] current_state, next_state;
    reg [$clog2(INPUT_SIZE):0] input_cnt;
    reg [$clog2(OUTPUT_SIZE)-1:0] output_cnt;
    reg signed [INPUT_SIZE*INPUT_DATA_WIDTH-1:0] i_spikes_reg;
    reg signed [OUTPUT_SIZE*OUTPUT_DATA_WIDTH-1:0] o_logits_reg;
    reg signed [ACC_WIDTH-1:0] accumulator_full;
    wire signed [INPUT_DATA_WIDTH-1:0] weight_data;
    wire signed [BIAS_WIDTH-1:0] bias_data;
    reg signed [INPUT_DATA_WIDTH-1:0] current_input_val_dly;
    wire signed [INPUT_DATA_WIDTH-1:0] current_input_val = i_spikes_reg[input_cnt*INPUT_DATA_WIDTH +: INPUT_DATA_WIDTH];
    wire signed [16:0] product = $signed(current_input_val_dly) * $signed(weight_data);

    wire is_last_mac_cycle  = (input_cnt == INPUT_SIZE - 1);
    wire is_last_neuron     = (output_cnt == OUTPUT_SIZE - 1);

    // --- ��������߼� ---

    // **������������**: ����һ��������ͨ������߼�����������������ۼӺ͡�
    wire signed [ACC_WIDTH-1:0] final_acc_val = accumulator_full + $signed(product);

    // **������������**: �޸� acc_with_bias ��������Դ���� FINISH ״̬ʹ����������ۼ�ֵ��
    wire signed [ACC_WIDTH-1:0] acc_with_bias = ((current_state == FSM_FINISH) ? final_acc_val : accumulator_full) + $signed(bias_data);

    wire signed [ACC_WIDTH-1:0] scaled_val_full = $signed(acc_with_bias) >>> LOGITS_SCALE_SHIFT;
    localparam SIGNED_16_MAX = 32767, SIGNED_16_MIN = -32768;
    wire signed [OUTPUT_DATA_WIDTH-1:0] clamped_output = (scaled_val_full > SIGNED_16_MAX) ? SIGNED_16_MAX : (scaled_val_full < SIGNED_16_MIN) ? SIGNED_16_MIN : scaled_val_full[OUTPUT_DATA_WIDTH-1:0];

    // --- ROM ʵ���� ---
    rom_sync #( .DATA_WIDTH (INPUT_DATA_WIDTH), .ADDR_WIDTH ($clog2(INPUT_SIZE * OUTPUT_SIZE)), .MEM_FILE (WEIGHT_ROM_FILE) ) weight_rom ( .clk(clk), .addr(output_cnt * INPUT_SIZE + input_cnt), .data_out(weight_data) );
    rom_sync #( .DATA_WIDTH (BIAS_WIDTH), .ADDR_WIDTH ($clog2(OUTPUT_SIZE)), .MEM_FILE (BIAS_ROM_FILE) ) bias_rom ( .clk(clk), .addr(output_cnt), .data_out(bias_data) );

    // --- FSM ״̬ת���߼� (�ޱ仯) ---
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

    // --- FSM ʱ���߼� ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= FSM_IDLE;
            input_cnt <= 0; output_cnt <= 0; accumulator_full <= 0; o_logits_reg <= 0; current_input_val_dly <= 0;
        end else begin
            current_state <= next_state;

            // **������������**: �Ƴ�ԭ����FSM_CALC��FINISH֮�������ۼ��������߼���
            // �������ͳһ��IDLE����CALCʱ���Լ�FINISH������һ��CALCʱ����

            case (current_state)
                FSM_IDLE:
                    if (i_valid) begin
                        i_spikes_reg <= i_spikes;
                        input_cnt    <= 0;
                        output_cnt   <= 0;
                        accumulator_full <= 0; // **������������**: ȷ��ÿ�ο�ʼ����ʱ�ۼ�������
                    end
                FSM_CALC: begin
                    current_input_val_dly <= current_input_val;
                    // **������������**: �򻯲������ۼ��߼���ʹ����dense_layer����һ��
                    if (input_cnt > 0) begin
                        accumulator_full <= accumulator_full + $signed(product);
                    end
                    input_cnt <= input_cnt + 1;
                end
                FSM_FINISH: begin
                    o_logits_reg[output_cnt*OUTPUT_DATA_WIDTH +: OUTPUT_DATA_WIDTH] <= clamped_output;
                    if (!is_last_neuron) begin
                        output_cnt <= output_cnt + 1;
                        input_cnt  <= 0;
                        accumulator_full <= 0; // **������������**: Ϊ��һ����Ԫ���������ۼ���
                    end
                end
                FSM_DONE: begin end
            endcase
        end
    end
    
    assign i_ready  = (current_state == FSM_IDLE);
    assign o_valid  = (current_state == FSM_DONE);
    assign o_logits = o_logits_reg;
endmodule