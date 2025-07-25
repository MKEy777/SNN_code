`timescale 1ns / 1ps

module spi_slave_rx #(
    parameter DATA_WIDTH      = 8,
    parameter FEATURE_LENGTH  = 300
)(
    input  wire clk,
    input  wire rst_n,
    input  wire sclk,
    input  wire mosi,
    input  wire cs,
    output wire o_valid, 
    input  wire i_ready,
    output wire signed [FEATURE_LENGTH*DATA_WIDTH-1:0] o_features
);

    reg signed [DATA_WIDTH-1:0] feature_buffer [0:FEATURE_LENGTH-1];
    reg [DATA_WIDTH-1:0]      shift_reg;
    reg [$clog2(DATA_WIDTH):0]      bit_count;
    reg [$clog2(FEATURE_LENGTH):0] byte_count;
    
    localparam FSM_IDLE      = 2'b00;
    localparam FSM_RECEIVING = 2'b01;
    localparam FSM_DONE      = 2'b10;

    reg [1:0] current_state, next_state;
    
    assign o_valid = (current_state == FSM_DONE);

    genvar g;
    generate
        for (g = 0; g < FEATURE_LENGTH; g = g + 1) begin
            assign o_features[(g*DATA_WIDTH) +: DATA_WIDTH] = feature_buffer[g];
        end
    endgenerate

    reg  sclk_d1, sclk_d2;
    wire sclk_falling_edge = sclk_d1 && !sclk_d2;

    always @(posedge clk) begin
        sclk_d1 <= sclk;
        sclk_d2 <= sclk_d1;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) current_state <= FSM_IDLE;
        else current_state <= next_state;
    end

    always @(*) begin
        next_state = current_state;
        case (current_state)
            FSM_IDLE:      if (!cs) next_state = FSM_RECEIVING;
            FSM_RECEIVING: begin
                if (byte_count == FEATURE_LENGTH) next_state = FSM_DONE;
                else if (cs) next_state = FSM_IDLE;
            end
            FSM_DONE:      if (i_ready) next_state = FSM_IDLE;
        endcase
    end
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bit_count <= 0;
            byte_count <= 0;
            shift_reg <= 0;
        end else begin
            case(current_state)
                FSM_IDLE: begin
                    bit_count <= 0;
                    byte_count <= 0;
                end
                
                FSM_RECEIVING: begin
                    if (sclk_falling_edge) begin
                        shift_reg <= {shift_reg[DATA_WIDTH-2:0], mosi};
                        if (bit_count == DATA_WIDTH-1) begin
                            bit_count <= 0;
                            feature_buffer[byte_count] <= {shift_reg, mosi};
                            byte_count <= byte_count + 1;
                        end else begin
                            bit_count <= bit_count + 1;
                        end
                    end
                end

                FSM_DONE: begin
                end
            endcase

            if (cs && current_state != FSM_DONE) begin
                 bit_count <= 0;
                 byte_count <= 0;
            end
        end
    end

    function integer clog2;
        input integer value;
        begin
            value = value - 1;
            for (clog2 = 0; value > 0; clog2 = clog2 + 1)
                value = value >> 1;
        end
    endfunction
endmodule