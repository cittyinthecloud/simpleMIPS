"""
This file implements tests of the circuits module using the Hypothesis module, to 
confirm that these gates work correctly for all inputs, as compared to more 
pythonic implementations
"""
import circuits
import pytest
from hypothesis import given, note, assume
from hypothesis.strategies import booleans, tuples
import numpy as np
from functools import reduce

# Basic Gates

def list_of_bits_to_int(bits):
    out = 0
    for bit in bits:
        out = out << 1 | bit
    return out

@given(booleans(), booleans())
def test_andgate(in0, in1):
    assert circuits.AndGate(in0, in1).getCircuitOutput() == (in0 and in1)


@given(booleans(), booleans())
def test_orgate(in0, in1):
    assert circuits.OrGate(in0, in1).getCircuitOutput() == (in0 or in1)


@given(booleans())
def test_notgate(in0):
    assert circuits.NotGate(in0).getCircuitOutput() == (not in0)


@given(booleans(), booleans())
def test_xorgate(in0, in1):
    assert circuits.XorGate(in0, in1).getCircuitOutput() == (bool(in0) != bool(in1))



@given(tuples(*(booleans() for _ in range(4))))
def test_orgate4(inputs):
    gate_out = circuits.OrGate4(*inputs).getCircuitOutput()
    assert gate_out == any(inputs)


@given(tuples(*(booleans() for _ in range(16))))
def test_orgate16(inputs):
    gate_out = circuits.OrGate16(*inputs).getCircuitOutput()
    assert gate_out == any(inputs)


@given(tuples(*(booleans() for _ in range(32))))
def test_orgate32(inputs):
    gate_out = circuits.OrGate32(*inputs).getCircuitOutput()
    assert gate_out == any(inputs)

@given(tuples(*(booleans() for _ in range(6))))
def test_andgate6(inputs):
    gate_out = circuits.AndGate6(*inputs).getCircuitOutput()
    assert gate_out == all(inputs)

@given(tuples(*(booleans() for _ in range(5))))
def test_andgate5(inputs):
    gate_out = circuits.AndGate5(*inputs).getCircuitOutput()
    assert gate_out == all(inputs)

# Multiplexers


@given(booleans(), booleans(), booleans())
def test_2to1mux(in0, in1, sel):
    mux_out = circuits.TwoToOneMux(in0, in1, sel).getCircuitOutput()
    if sel:
        assert mux_out == in1
    else:
        assert mux_out == in0


@given(booleans(), booleans(), booleans(), booleans(), booleans(), booleans())
def test_4to1mux(in0, in1, in2, in3, sel0, sel1):
    mux_out = circuits.FourToOneMux(in0, in1, in2, in3, sel0, sel1).getCircuitOutput()

    if sel1:
        if sel0:
            assert mux_out == in3
        else:
            assert mux_out == in2
    else:
        if sel0:
            assert mux_out == in1
        else:
            assert mux_out == in0


@given(booleans(), booleans(), booleans())
def test_fulladder(in0, in1, carry_in):
    adder_out = circuits.FullAdder(in0, in1, carry_in).getCircuitOutput()

    sum = in0 + in1 + carry_in

    expected_output = {
        0: circuits.FullAdderOutput(0, 0),
        1: circuits.FullAdderOutput(1, 0),
        2: circuits.FullAdderOutput(0, 1),
        3: circuits.FullAdderOutput(1, 1),
    }[sum]

    assert adder_out == expected_output


alu_control_states = [
    # ((aluop1, aluop0, f5, f4, f3, f2, f1, f0), (op3, op2, op1, op0))
    ((0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 1, 0)),
    ((0, 1, 0, 0, 0, 0, 0, 0), (0, 1, 1, 0)),
    ((1, 0, 0, 0, 0, 0, 0, 0), (0, 0, 1, 0)),
    ((1, 0, 0, 0, 0, 0, 1, 0), (0, 1, 1, 0)),
    ((1, 0, 0, 0, 0, 1, 0, 0), (0, 0, 0, 0)),
    ((1, 0, 0, 0, 0, 1, 0, 1), (0, 0, 0, 1)),
    ((1, 0, 0, 0, 1, 0, 1, 0), (0, 1, 1, 1)),
]


@pytest.mark.parametrize("input, output", alu_control_states)
def test_alu_control(input, output):
    assert circuits.ALUControl(*input).getCircuitOutput() == output


@given(
    booleans(),
    booleans(),
    booleans(),
    booleans(),
    booleans(),
    booleans(),
    booleans(),
    booleans(),
)
def test_1bit_alu(a, b, carry_in, less, a_invert, b_invert, op1, op0):
    alu_out = circuits.ALU_1bit(
        a, b, carry_in, less, a_invert, b_invert, op1, op0
    ).getCircuitOutput()

    operation = op1 * 2 + op0

    modifiedA = (not a) if a_invert else a
    modifiedB = (not b) if b_invert else b

    real_sum = modifiedA + modifiedB + carry_in

    sum = bool(real_sum & 0b01)
    carry_out = bool(real_sum & 0b10)

    # Check result
    if operation == 0:
        assert alu_out.result == (modifiedA and modifiedB)
    elif operation == 1:
        assert alu_out.result == (modifiedA or modifiedB)
    elif operation == 2:
        assert alu_out.result == sum
    elif operation == 3:
        assert alu_out.result == less

    # Check set
    assert alu_out.set == sum

    # Check carry out
    assert alu_out.carry_out == carry_out

    # Check overflow
    assert alu_out.overflow == (carry_out != carry_in)

@given(
    a = tuples(*(booleans() for _ in range(32))), # A is a 32-bit input
    b = tuples(*(booleans() for _ in range(32))), # B is a 32-bit input
    a_invert = booleans(),
    b_negate = booleans(),
    op1 = booleans(),
    op0 = booleans()
)

@pytest.mark.filterwarnings("ignore: overflow encountered in scalar add") # NumPy yells on a detected overflow, but we're depending on that overflow
def test_alu_32bit(a, b, a_invert, b_negate, op1, op0):
    assume((not b_negate) or (op1 and not op0)) # Invalid input state.
    assume(not (op1 and op0) or (not (a_invert or b_negate)))
    alu_out = circuits.ALU_32Bit(a, b, a_invert, b_negate, op1, op0).getCircuitOutput()

    # Turn the bit lists into numpy int32s
    a_int = np.array(list_of_bits_to_int(a)).astype(np.int32)
    b_int = np.array(list_of_bits_to_int(b)).astype(np.int32)
    note("A= " + repr(a_int))
    note("B= " + repr(b_int))

    if a_invert:
        a_int = ~a_int
    
    if b_negate:
        b_int = -b_int
     
    operation = op1 * 2 + op0
    

    note("ALU output: "+repr(alu_out))
    result_int = np.array(int("".join(str(int(bit)) for bit in alu_out.result), 2)).astype(np.int32)
    note("Result= "+repr(result_int))
    if operation == 0:
        assert a_int&b_int == result_int
    if operation == 1:
        assert a_int|b_int == result_int
    if operation == 2:
        assert a_int+b_int == result_int
    if operation == 3:
        pass
    

@given(tuples(*(booleans() for _ in range(32))))
def test_instruction_splitter(instruction):
    instruction = list(instruction)
    decoder_out = circuits.InstructionSplitter(instruction).getCircuitOutput()
    
    # Assert that everything is the correct length
    assert len(decoder_out.funct) == 6
    assert len(decoder_out.shamt) == 5
    assert len(decoder_out.rd) == 5
    assert len(decoder_out.rt) == 5
    assert len(decoder_out.rs) == 5
    assert len(decoder_out.opcode) == 6
    assert len(decoder_out.immediate) == 16

    # Assert that everything for R-type instructions was split out correctly
    assert decoder_out.opcode == instruction[:6]
    assert decoder_out.rs == instruction[6:11]
    assert decoder_out.rt == instruction[11:16]
    assert decoder_out.rd == instruction[16:21]
    assert decoder_out.shamt == instruction[21:26]
    assert decoder_out.funct == instruction[26:32]


@given(tuples(*(booleans() for _ in range(5))))
def test_decoder(inputs):
    decoder_out = circuits.Decoder(*inputs).getCircuitOutput()
    assert sum(decoder_out) == 1 # Only one output should ever be true

    i = reduce(lambda a, b: (a<<1) + int(b), inputs)
    
    assert decoder_out[i] == 1 # Correct output should be true