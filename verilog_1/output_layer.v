`timescale 1ns / 1ps

// ============================================================================
// == output_layer (�Ż���)
// ============================================================================
module output_layer #(
    parameter INPUT_SIZE        = 128,
    parameter OUTPUT_SIZE       = 3,
    parameter DATA_WIDTH        = 32,
    parameter FRACTIONAL_BITS   = 16,
    parameter WEIGHT_ROM_FILE   = "output_w_rom.mem",
    parameter BIAS_ROM_FILE     = "output_d_rom.mem"
)(
    input  wire                           clk,
    input  wire                           rst_n,
    input  wire                           i_valid,
    output wire                           i_ready,
    output wire                           o_valid,
    input  wire                           o_ready,
    input  wire signed [INPUT_SIZE*DATA_WIDTH-1:0]   i_spikes,
    output wire signed [OUTPUT_SIZE*DATA_WIDTH-1:0]  o_logits
);

    // [�Ż�] FSM �� 5 ״̬��Ϊ 3 ״̬���ϲ���ԭ FINISH, LATCH, HOLD ״̬
    localparam FSM_IDLE   = 2'd0;
    localparam FSM_CALC   = 2'd1;
    localparam FSM_DONE   = 2'd2;
    reg [1:0] current_state, next_state;

    // �ڲ��Ĵ������ź� (o_valid_reg ���֣���ʵ���Ƚ�������)
    reg [$clog2(INPUT_SIZE)-1:0]  input_cnt;
    reg [$clog2(OUTPUT_SIZE)-1:0] output_cnt;
    reg signed [INPUT_SIZE*DATA_WIDTH-1:0]  i_spikes_reg;
    reg signed [OUTPUT_SIZE*DATA_WIDTH-1:0] o_logits_reg;
    reg signed [DATA_WIDTH-1:0] logit_final_d1;
    reg                                     o_valid_reg; // ���� o_valid �Ĵ���
    reg signed [DATA_WIDTH-1:0] tj_shifted_p1;
    localparam FULL_ACC_WIDTH = 2 * DATA_WIDTH + $clog2(INPUT_SIZE);
    reg signed [FULL_ACC_WIDTH-1:0] accumulator_full;
    localparam signed [DATA_WIDTH-1:0] T_MIN = 32'h00030000;
    wire signed [DATA_WIDTH-1:0] weight_data;
    wire signed [DATA_WIDTH-1:0] bias_data;
    wire is_last_mac    = (input_cnt == INPUT_SIZE - 1);
    wire is_last_neuron = (output_cnt == OUTPUT_SIZE - 1);
    reg  is_last_mac_d1;
    wire signed [DATA_WIDTH-1:0] current_tj = i_spikes_reg[input_cnt*DATA_WIDTH +: DATA_WIDTH];
    wire signed [DATA_WIDTH-1:0] tj_shifted = T_MIN - current_tj;
    wire signed [2*DATA_WIDTH-1:0] product = tj_shifted_p1 * weight_data;
    wire signed [FULL_ACC_WIDTH-1:0] final_accumulator = accumulator_full + $signed(product);
    wire signed [DATA_WIDTH-1:0] sum_scaled_down = $signed(final_accumulator >>> FRACTIONAL_BITS);
    wire signed [DATA_WIDTH-1:0] logit_final = sum_scaled_down + bias_data;

    // ROM ʵ���� (�ޱ仯)
    rom_sync #( .DATA_WIDTH (DATA_WIDTH), .ADDR_WIDTH ($clog2(INPUT_SIZE * OUTPUT_SIZE)), .MEM_FILE(WEIGHT_ROM_FILE) ) weight_rom_inst ( .clk(clk), .addr(output_cnt * INPUT_SIZE + input_cnt), .data_out(weight_data) );
    rom_sync #( .DATA_WIDTH (DATA_WIDTH), .ADDR_WIDTH ($clog2(OUTPUT_SIZE)), .MEM_FILE(BIAS_ROM_FILE) ) bias_rom_inst ( .clk(clk), .addr(output_cnt), .data_out(bias_data) );

    // --- [�Ż�] ״̬ת���߼� ---
    always @(*) begin
        next_state = current_state;
        case (current_state)
            FSM_IDLE:   if (i_valid) next_state = FSM_CALC;
            FSM_CALC:   if (is_last_mac_d1) next_state = FSM_DONE;
            FSM_DONE: begin
                if (is_last_neuron) begin
                    // ��������һ����Ԫ���ȴ����ν���
                    if (o_ready) next_state = FSM_IDLE;
                end else begin
                    // ����ֱ�ӿ�ʼ������һ����Ԫ
                    next_state = FSM_CALC;
                end
            end
        endcase
    end

    // --- ʱ���߼� ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state       <= FSM_IDLE;
            input_cnt           <= 0;
            output_cnt          <= 0;
            accumulator_full    <= 0;
            tj_shifted_p1       <= 0;
            o_logits_reg        <= 0;
            logit_final_d1      <= 0;
            is_last_mac_d1      <= 1'b0;
            o_valid_reg         <= 1'b0;
        end else begin
            logit_final_d1 <= logit_final;
            is_last_mac_d1 <= is_last_mac;
            tj_shifted_p1  <= tj_shifted;
            current_state  <= next_state;
            
            case (current_state)
                FSM_IDLE: begin
                    o_valid_reg <= 1'b0;
                    if (i_valid) begin
                        i_spikes_reg     <= i_spikes;
                        input_cnt        <= 0;
                        output_cnt       <= 0;
                        accumulator_full <= 0;
                    end
                end

                FSM_CALC: begin
                    // [ά����״] �ۼ����߼����ֲ��䣬��ƥ������·������ˮ��ʱ��
                    if (input_cnt == 1)
                        accumulator_full <= $signed(product);
                    else if (input_cnt > 1)
                        accumulator_full <= accumulator_full + $signed(product);
                    
                    input_cnt <= input_cnt + 1;
                end

                // --- [�Ż�] �µ� DONE ״̬ ---
                FSM_DONE: begin
                    o_logits_reg[output_cnt*DATA_WIDTH +: DATA_WIDTH] <= logit_final_d1;
                    if (!is_last_neuron) begin
                        // ���������ԪҪ���㣬��λ������
                        output_cnt       <= output_cnt + 1;
                        input_cnt        <= 0;
                        accumulator_full <= 0;
                    end else begin
                        // ��������һ����Ԫ����λ o_valid_reg
                        o_valid_reg <= 1'b1;
                    end
                    
                    // �����ν������ݺ����� o_valid_reg
                    if (o_valid_reg && o_ready) begin
                        o_valid_reg <= 1'b0;
                    end
                end
            endcase
        end
    end

    // ����ź�
    assign i_ready  = (current_state == FSM_IDLE);
    assign o_valid  = o_valid_reg;
    assign o_logits = o_logits_reg;

endmodule