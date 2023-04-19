__author__ = "Your names"
__copyright__ = "Copyright @2023"

from collections import namedtuple
from typing import Any, Iterable, Sequence

# Two inputs is the common case.
class TwoInputCircuit:
    def __init__(self, in0: bool, in1: bool) -> None:
        self.in0 = in0
        self.in1 = in1

    def getCircuitOutput(self):
        # How did we get here?
        raise NotImplementedError


# Basic Gates


class AndGate(TwoInputCircuit):
    def getCircuitOutput(self) -> bool:
        return self.in0 and self.in1


class OrGate(TwoInputCircuit):
    def getCircuitOutput(self) -> bool:
        return self.in0 or self.in1


class NotGate:
    def __init__(self, in0: bool) -> None:
        self.in0 = in0

    def getCircuitOutput(self) -> bool:
        return not self.in0


class XorGate(TwoInputCircuit):
    """
    XOR gate constructed from NOT, AND, and OR gates.

    A^B = (!A&B) | (!B&A)
    """

    def getCircuitOutput(self):
        return OrGate(
            AndGate(
                NotGate(self.in0).getCircuitOutput(),
                self.in1,
            ).getCircuitOutput(),
            AndGate(
                NotGate(self.in1).getCircuitOutput(),
                self.in0,
            ).getCircuitOutput(),
        ).getCircuitOutput()


class OrGate4:
    """
    4-input AND gate constructed from 3 2-input AND gates
    """

    def __init__(self, in0, in1, in2, in3):
        self.in0 = in0
        self.in1 = in1
        self.in2 = in2
        self.in3 = in3

    def getCircuitOutput(self):
        return OrGate(
            OrGate(
                self.in0,
                self.in1,
            ).getCircuitOutput(),
            OrGate(
                self.in2,
                self.in3,
            ).getCircuitOutput(),
        ).getCircuitOutput()


class OrGate16:
    """
    16-input AND gate constructed from 5 4-input AND gates
    """

    def __init__(
        self,
        in0,
        in1,
        in2,
        in3,
        in4,
        in5,
        in6,
        in7,
        in8,
        in9,
        in10,
        in11,
        in12,
        in13,
        in14,
        in15,
    ) -> None:
        self.in0 = in0
        self.in1 = in1
        self.in2 = in2
        self.in3 = in3
        self.in4 = in4
        self.in5 = in5
        self.in6 = in6
        self.in7 = in7
        self.in8 = in8
        self.in9 = in9
        self.in10 = in10
        self.in11 = in11
        self.in12 = in12
        self.in13 = in13
        self.in14 = in14
        self.in15 = in15

    def getCircuitOutput(self):
        return OrGate4(
            OrGate4(self.in0, self.in1, self.in2, self.in3).getCircuitOutput(),
            OrGate4(
                self.in4,
                self.in5,
                self.in6,
                self.in7,
            ).getCircuitOutput(),
            OrGate4(self.in8, self.in9, self.in10, self.in11).getCircuitOutput(),
            OrGate4(self.in12, self.in13, self.in14, self.in15).getCircuitOutput(),
        ).getCircuitOutput()


class OrGate32:
    """
    32-input AND gate constructed from 2 16 input and gates and a 2 input and gate
    """

    def __init__(
        self,
        in0,
        in1,
        in2,
        in3,
        in4,
        in5,
        in6,
        in7,
        in8,
        in9,
        in10,
        in11,
        in12,
        in13,
        in14,
        in15,
        in16,
        in17,
        in18,
        in19,
        in20,
        in21,
        in22,
        in23,
        in24,
        in25,
        in26,
        in27,
        in28,
        in29,
        in30,
        in31,
    ) -> None:
        self.in0 = in0
        self.in1 = in1
        self.in2 = in2
        self.in3 = in3
        self.in4 = in4
        self.in5 = in5
        self.in6 = in6
        self.in7 = in7
        self.in8 = in8
        self.in9 = in9
        self.in10 = in10
        self.in11 = in11
        self.in12 = in12
        self.in13 = in13
        self.in14 = in14
        self.in15 = in15
        self.in16 = in16
        self.in17 = in17
        self.in18 = in18
        self.in19 = in19
        self.in20 = in20
        self.in21 = in21
        self.in22 = in22
        self.in23 = in23
        self.in24 = in24
        self.in25 = in25
        self.in26 = in26
        self.in27 = in27
        self.in28 = in28
        self.in29 = in29
        self.in30 = in30
        self.in31 = in31

    def getCircuitOutput(self):
        return OrGate(
            OrGate16(
                self.in0,
                self.in1,
                self.in2,
                self.in3,
                self.in4,
                self.in5,
                self.in6,
                self.in7,
                self.in8,
                self.in9,
                self.in10,
                self.in11,
                self.in12,
                self.in13,
                self.in14,
                self.in15,
            ).getCircuitOutput(),
            OrGate16(
                self.in16,
                self.in17,
                self.in18,
                self.in19,
                self.in20,
                self.in21,
                self.in22,
                self.in23,
                self.in24,
                self.in25,
                self.in26,
                self.in27,
                self.in28,
                self.in29,
                self.in30,
                self.in31,
            ).getCircuitOutput(),
        ).getCircuitOutput()


class AndGate6:
    """
    6-input AND gate creaed from 4 2-input and gates
    """

    def __init__(self, in0, in1, in2, in3, in4, in5):
        self.in0 = in0
        self.in1 = in1
        self.in2 = in2
        self.in3 = in3
        self.in4 = in4
        self.in5 = in5

    def getCircuitOutput(self):
        return AndGate(
            AndGate(
                AndGate(self.in0, self.in1).getCircuitOutput(),
                AndGate(self.in2, self.in3).getCircuitOutput(),
            ).getCircuitOutput(),
            AndGate(self.in4, self.in5).getCircuitOutput(),
        ).getCircuitOutput()


class AndGate5(AndGate6):
    """
    5-input AND gate constructed from a 6 input AND gate with the 6th input
    forced to be true
    """

    def __init__(self, in0, in1, in2, in3, in4):
        self.in0 = in0
        self.in1 = in1
        self.in2 = in2
        self.in3 = in3
        self.in4 = in4
        self.in5 = True


# Multiplexers


class TwoToOneMux:
    """
    Implementation of a two to one mux using simple gates.
    """

    def __init__(self, in0, in1, sel) -> None:
        self.in0 = in0
        self.in1 = in1
        self.sel = sel

    def getCircuitOutput(self):
        return OrGate(
            AndGate(self.in0, NotGate(self.sel).getCircuitOutput()).getCircuitOutput(),
            AndGate(self.in1, self.sel).getCircuitOutput(),
        ).getCircuitOutput()


class FourToOneMux:
    """
    Implementation of a four to one mux using three two to one muxs.
    """

    def __init__(self, in0, in1, in2, in3, sel0, sel1) -> None:
        self.in0 = in0
        self.in1 = in1
        self.in2 = in2
        self.in3 = in3
        self.sel0 = sel0
        self.sel1 = sel1

    def getCircuitOutput(self):
        return TwoToOneMux(
            TwoToOneMux(self.in0, self.in1, self.sel0).getCircuitOutput(),
            TwoToOneMux(self.in2, self.in3, self.sel0).getCircuitOutput(),
            self.sel1,
        ).getCircuitOutput()


# ALU

FullAdderOutput = namedtuple("FullAdderOutput", ["sum", "carry_out"])


class FullAdder:
    def __init__(self, in0, in1, carry_in) -> None:
        self.in0 = in0
        self.in1 = in1
        self.carry_in = carry_in

    def getCircuitOutput(self):
        xor_inputs = XorGate(self.in0, self.in1)

        sum = XorGate(xor_inputs.getCircuitOutput(), self.carry_in).getCircuitOutput()

        carry = OrGate(
            AndGate(self.in0, self.in1).getCircuitOutput(),
            AndGate(xor_inputs.getCircuitOutput(), self.carry_in).getCircuitOutput(),
        ).getCircuitOutput()

        return FullAdderOutput(sum, carry)


class ALUControl:
    """
    ALU Control circuit. Output is in most to least significant order.
    """

    def __init__(self, aluOp1, aluOp0, f5, f4, f3, f2, f1, f0) -> None:
        self.aluOp0 = aluOp0
        self.aluOp1 = aluOp1
        self.f5 = f5
        self.f4 = f4
        self.f3 = f3
        self.f2 = f2
        self.f1 = f1
        self.f0 = f0

    def getCircuitOutput(self):
        op2 = OrGate(
            self.aluOp0, AndGate(self.aluOp1, self.f1).getCircuitOutput()
        ).getCircuitOutput()

        op1 = OrGate(
            NotGate(self.aluOp1).getCircuitOutput(), NotGate(self.f2).getCircuitOutput()
        ).getCircuitOutput()

        op0 = AndGate(
            self.aluOp1, OrGate(self.f3, self.f0).getCircuitOutput()
        ).getCircuitOutput()

        return (0, op2, op1, op0)


OneBitALUOutput = namedtuple(
    "ALUOutput_1bit", ["result", "carry_out", "set", "overflow"]
)


class ALU_1bit:
    def __init__(self, a, b, carry_in, less, a_invert, b_invert, op1, op0) -> None:
        self.a = a
        self.b = b
        self.carry_in = carry_in
        self.a_invert = a_invert
        self.b_invert = b_invert
        self.op1 = op1
        self.op0 = op0
        self.less = less

    def getCircuitOutput(self):
        modifiedA = TwoToOneMux(
            self.a,
            NotGate(self.a).getCircuitOutput(),
            self.a_invert,
        ).getCircuitOutput()

        modifiedB = TwoToOneMux(
            self.b, NotGate(self.b).getCircuitOutput(), self.b_invert
        ).getCircuitOutput()

        and_ = AndGate(modifiedA, modifiedB).getCircuitOutput()

        or_ = OrGate(modifiedA, modifiedB).getCircuitOutput()

        add = FullAdder(modifiedA, modifiedB, self.carry_in).getCircuitOutput()

        result = FourToOneMux(
            and_, or_, add.sum, self.less, self.op0, self.op1
        ).getCircuitOutput()

        overflow = XorGate(self.carry_in, add.carry_out).getCircuitOutput()

        return OneBitALUOutput(result, add.carry_out, add.sum, overflow)


ALUOutput = namedtuple("ALUOutput", ["result", "zero", "overflow"])


class ALU_32Bit:
    """
    A 32-bit ALU constructed from 32 one bit ALUs
    """

    def __init__(
        self,
        a: Sequence,
        b: Sequence,
        a_invert: bool,
        b_negate: bool,
        op1: bool,
        op0: bool,
    ) -> None:
        self.a = a
        self.b = b
        self.a_invert = a_invert
        self.b_negate = b_negate
        self.op1 = op1
        self.op0 = op0

    def getCircuitOutput(self):
        alus: Sequence[ALU_1bit] = []
        for i in range(32):

            a_bit = self.a[~i]
            b_bit = self.b[~i]
            # if self.b_negate:
            #     b_bit = not b_bit

            if i == 0:  # first alu
                carry_in = self.b_negate
            else:
                carry_in = alus[-1].getCircuitOutput().carry_out

            alus.append(
                ALU_1bit(
                    a_bit,
                    b_bit,
                    carry_in,
                    False,
                    self.a_invert,
                    self.b_negate,
                    self.op1,
                    self.op0,
                )
            )

            # if i == 0:
            #     print(
            #         a_bit,
            #         b_bit,
            #         carry_in,
            #         False,
            #         self.a_invert,
            #         self.b_negate,
            #         self.op1,
            #         self.op0,
            #         alus[-1].getCircuitOutput()
            #     )

        # Less input of the first ALU needs to the set output of the last ALU
        alus[0].less = alus[-1].getCircuitOutput().set

        # result is a list of the 32 "result" outputs of the alus, in MSB->LSB order.
        result = []
        for alu in reversed(alus):
            result.append(alu.getCircuitOutput().result)

        # Calculate the zero flag
        zero = NotGate(OrGate32(*result).getCircuitOutput()).getCircuitOutput()

        return ALUOutput(result, zero, alus[-1].getCircuitOutput().overflow)


ControlOutput = namedtuple(
    "ControlOutput",
    [
        "reg_dst",
        "alusrc",
        "memtoreg",
        "regwrite",
        "memread",
        "memwrite",
        "branch",
        "aluOp1",
        "aluOp0",
    ],
)


class Control:
    def __init__(self, op5, op4, op3, op2, op1, op0) -> None:
        self.op5 = op5
        self.op4 = op4
        self.op3 = op3
        self.op2 = op2
        self.op1 = op1
        self.op0 = op0

    def getCircuitOutput(self):
        r_format = AndGate6(
            NotGate(self.op5).getCircuitOutput(),
            NotGate(self.op4).getCircuitOutput(),
            NotGate(self.op3).getCircuitOutput(),
            NotGate(self.op2).getCircuitOutput(),
            NotGate(self.op1).getCircuitOutput(),
            NotGate(self.op0).getCircuitOutput(),
        ).getCircuitOutput()

        lw = AndGate6(
            self.op5,
            NotGate(self.op4).getCircuitOutput(),
            NotGate(self.op3).getCircuitOutput(),
            NotGate(self.op2).getCircuitOutput(),
            self.op1,
            self.op0,
        ).getCircuitOutput()

        sw = AndGate6(
            self.op5,
            NotGate(self.op4).getCircuitOutput(),
            self.op3,
            NotGate(self.op2).getCircuitOutput(),
            self.op1,
            self.op0,
        ).getCircuitOutput()

        beq = AndGate6(
            NotGate(self.op5).getCircuitOutput(),
            NotGate(self.op4).getCircuitOutput(),
            NotGate(self.op3).getCircuitOutput(),
            self.op2,
            NotGate(self.op1).getCircuitOutput(),
            NotGate(self.op0).getCircuitOutput(),
        ).getCircuitOutput()

        return ControlOutput(
            r_format,
            OrGate(lw, sw).getCircuitOutput(),
            lw,
            OrGate(r_format, lw).getCircuitOutput(),
            lw,
            sw,
            beq,
            r_format,
            beq,
        )


class Registers:
    def __init__(self) -> None:
        self._registers = [[0 for _ in range(32)] for _ in range(32)]

    def getRegister(self, i):
        return self._registers[i]

    def setRegister(self, i, reg):
        self._registers[i] = reg

    def getAllRegValues(self):
        values = []
        for register in self._registers:
            value = ""

            for bit in register:
                value += "1" if bit else "0"

            values.append(value)
        return values


InstructionSplitterOutput = namedtuple(
    "InstructionSplitterOutput",
    ["immediate", "rd", "rt", "rs", "opcode", "shamt", "funct"],
)


class InstructionSplitter:
    def __init__(self, instruction: Iterable[bool]) -> None:
        """
        instruction in list form msb->lsb order
        """
        self.inst = list(instruction)
        self.inst.reverse()  # This makes more sense to store lsb->msb internally

    def getCircuitOutput(self):
        immediate = self.inst[0:16]
        rd = self.inst[11:16]
        rt = self.inst[16:21]
        rs = self.inst[21:26]
        opcode = self.inst[26:]
        shamt = self.inst[6:11]
        funct = self.inst[0:6]

        immediate.reverse()
        rd.reverse()
        rt.reverse()
        rs.reverse()
        opcode.reverse()
        shamt.reverse()
        funct.reverse()

        return InstructionSplitterOutput(immediate, rd, rt, rs, opcode, shamt, funct)


class Decoder:
    def __init__(self, in5, in4, in3, in2, in1) -> None:
        self.in5 = in5
        self.in4 = in4
        self.in3 = in3
        self.in2 = in2
        self.in1 = in1

    def getCircuitOutput(self):
        """
        Output is such that the index into the iterable is the number
        represented by the input
        """
        not_in5 = NotGate(self.in5).getCircuitOutput()
        not_in4 = NotGate(self.in4).getCircuitOutput()
        not_in3 = NotGate(self.in3).getCircuitOutput()
        not_in2 = NotGate(self.in2).getCircuitOutput()
        not_in1 = NotGate(self.in1).getCircuitOutput()
        in5 = self.in5
        in4 = self.in4
        in3 = self.in3
        in2 = self.in2
        in1 = self.in1

        return (
            AndGate5(not_in5, not_in4, not_in3, not_in2, not_in1).getCircuitOutput(),
            AndGate5(not_in5, not_in4, not_in3, not_in2, in1).getCircuitOutput(),
            AndGate5(not_in5, not_in4, not_in3, in2, not_in1).getCircuitOutput(),
            AndGate5(not_in5, not_in4, not_in3, in2, in1).getCircuitOutput(),
            AndGate5(not_in5, not_in4, in3, not_in2, not_in1).getCircuitOutput(),
            AndGate5(not_in5, not_in4, in3, not_in2, in1).getCircuitOutput(),
            AndGate5(not_in5, not_in4, in3, in2, not_in1).getCircuitOutput(),
            AndGate5(not_in5, not_in4, in3, in2, in1).getCircuitOutput(),
            AndGate5(not_in5, in4, not_in3, not_in2, not_in1).getCircuitOutput(),
            AndGate5(not_in5, in4, not_in3, not_in2, in1).getCircuitOutput(),
            AndGate5(not_in5, in4, not_in3, in2, not_in1).getCircuitOutput(),
            AndGate5(not_in5, in4, not_in3, in2, in1).getCircuitOutput(),
            AndGate5(not_in5, in4, in3, not_in2, not_in1).getCircuitOutput(),
            AndGate5(not_in5, in4, in3, not_in2, in1).getCircuitOutput(),
            AndGate5(not_in5, in4, in3, in2, not_in1).getCircuitOutput(),
            AndGate5(not_in5, in4, in3, in2, in1).getCircuitOutput(),
            AndGate5(in5, not_in4, not_in3, not_in2, not_in1).getCircuitOutput(),
            AndGate5(in5, not_in4, not_in3, not_in2, in1).getCircuitOutput(),
            AndGate5(in5, not_in4, not_in3, in2, not_in1).getCircuitOutput(),
            AndGate5(in5, not_in4, not_in3, in2, in1).getCircuitOutput(),
            AndGate5(in5, not_in4, in3, not_in2, not_in1).getCircuitOutput(),
            AndGate5(in5, not_in4, in3, not_in2, in1).getCircuitOutput(),
            AndGate5(in5, not_in4, in3, in2, not_in1).getCircuitOutput(),
            AndGate5(in5, not_in4, in3, in2, in1).getCircuitOutput(),
            AndGate5(in5, in4, not_in3, not_in2, not_in1).getCircuitOutput(),
            AndGate5(in5, in4, not_in3, not_in2, in1).getCircuitOutput(),
            AndGate5(in5, in4, not_in3, in2, not_in1).getCircuitOutput(),
            AndGate5(in5, in4, not_in3, in2, in1).getCircuitOutput(),
            AndGate5(in5, in4, in3, not_in2, not_in1).getCircuitOutput(),
            AndGate5(in5, in4, in3, not_in2, in1).getCircuitOutput(),
            AndGate5(in5, in4, in3, in2, not_in1).getCircuitOutput(),
            AndGate5(in5, in4, in3, in2, in1).getCircuitOutput(),
        )


class SimpleMIPS:
    def __init__(self) -> None:
        self.instruction = [0 for _ in range(32)]
        self.registers = Registers()

    def tick(self):
        splitter = InstructionSplitter(self.instruction).getCircuitOutput()

        control = Control(*splitter.opcode).getCircuitOutput()

        alu_control = ALUControl(
            control.aluOp1, control.aluOp0, *splitter.funct
        ).getCircuitOutput()

        rs_int = int("".join("1" if x else "0" for x in splitter.rs), 2)
        rs_content = self.registers.getRegister(rs_int)

        rt_int = int("".join("1" if x else "0" for x in splitter.rt), 2)
        rt_content = self.registers.getRegister(rt_int)

        alu = ALU_32Bit(rs_content, rt_content, *alu_control).getCircuitOutput()

        rd_int = int("".join("1" if x else "0" for x in splitter.rd), 2)
        self.registers.setRegister(rd_int, alu.result)
